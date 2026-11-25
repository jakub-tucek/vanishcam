"""Run gesture recognition on a background thread so it overlaps rendering and display.

MediaPipe and OpenCV release the GIL, so the detectors really run in parallel with the
main loop. The main loop keeps every decision (hold gates, starting effects, HUD); the
worker only turns a frame into observations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Observation:
    mode: str
    value: Any        # True/False, or (gesture name, score) for the V sign.
    detail: str       # Preview hint, e.g. "Arms crossed: yes   at chest height: no".
    debug: str = ""   # --gesture-debug line without the [Effect] prefix, if requested.


@dataclass(frozen=True)
class GestureJob:
    image: Any                       # mp.Image of the camera frame.
    width: int
    height: int
    now: float                       # Frame time, for the hold gates.
    phase: str                       # Phase when submitted; stale results are dropped.
    modes: tuple[str, ...]           # Effects due for a check.
    recognizers: dict[str, Any]
    hand_source: Any = None          # Shared hand landmarker, if several hand effects are on.
    v_retry_views: int = 3
    debug: bool = False              # Build the --gesture-debug text (costs a little).


class GestureWorker:
    """One job at a time; a frame arriving while busy is skipped, never queued."""

    def __init__(self, observe: Callable[[GestureJob], list[Observation]], threaded: bool = True):
        self._observe = observe
        self._threaded = threaded
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job: GestureJob | None = None
        self._result: tuple[GestureJob, list[Observation]] | None = None
        self._error: BaseException | None = None
        self._busy = False
        self._closed = False
        self._thread: threading.Thread | None = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="gestures", daemon=True)
            self._thread.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, job: GestureJob) -> bool:
        if not self._threaded:
            self._result = (job, self._observe(job))
            return True
        with self._lock:
            if self._busy or self._closed:
                return False
            self._busy = True
            self._job = job
        self._wake.set()
        return True

    def poll(self) -> tuple[GestureJob, list[Observation]] | None:
        """The finished job and its observations, once; re-raises a worker failure."""
        with self._lock:
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            result, self._result = self._result, None
        return result

    def close(self) -> None:
        """Stop and wait, so no recognizer is in use when the models are closed."""
        with self._lock:
            self._closed = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                if self._closed:
                    return
                job = self._job
            try:
                observations = self._observe(job)
            except BaseException as error:  # Surfaced in the main loop by poll().
                with self._lock:
                    self._error = error
                    self._busy = False
                continue
            with self._lock:
                self._result = (job, observations)
                self._busy = False
