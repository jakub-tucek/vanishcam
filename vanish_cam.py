"""VanishCam: gesture-triggered disappearance effects for a webcam and virtual camera."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from contextlib import ExitStack
from pathlib import Path

if __name__ == "__main__":
    from mediapipe_log_filter import needs_filter, run_local_filtered
    if needs_filter(sys.argv[1:]):
        raise SystemExit(run_local_filtered(Path(__file__).resolve(), sys.argv[1:]))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from itachi_effect import (
    CROW_SHEET_PATH, SHARINGAN_PATH, EFFECT_SECONDS as ITACHI_SECONDS, ItachiCrowEffect,
    ItachiGestureGate, ItachiPoseRecognizer, load_crow_sprites, load_sharingan_art,
)
from goku_effect import (
    GokuGestureGate, GokuPoseRecognizer, GokuTeleport, HOLD_SECONDS,
    TELEPORT_SECONDS,
)
from particles_effect import PARTICLE_SECONDS, ParticleEffect
from naruto_effect import (
    EFFECT_SECONDS, LOG_PATH, joined_hands_from_landmarks, landed_log_anchor,
    load_log, overlay_log, render_substitution, substitution_anchor,
)
from vanish_sound import SOUND_PATH, TELEPORT_SOUND_PATH, VanishSound


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "Webcam - Preview"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
ALL_MODES = ("normal", "naruto", "particles", "goku", "itachi")

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
    FACE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
}

MODE_LABELS = {
    "normal": ("Normal", "V sign  /  dissolve"),
    "naruto": ("Naruto", "Palms together  /  log and smoke"),
    "particles": ("Particles", "V sign  /  particles in the wind"),
    "goku": ("Goku", "Two fingers on forehead  /  teleport"),
    "itachi": ("Itachi", "Hands at chest  /  flock of crows"),
}
GESTURE_HINTS = {
    "normal": "Show a V sign to vanish",
    "particles": "Show a V sign to vanish",
    "naruto": "Press your palms together to vanish",
    "goku": "Hold two fingers on your forehead",
    "itachi": "Touch your hands at your chest",
}
SETTINGS_PATH = (Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
                 / "vanishcam" / "settings.json")

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
VANISH_SECONDS = 3.6
GESTURE_INTERVAL_SECONDS = 0.14


@dataclass
class AppState:
    phase: str = "calibrating"
    calibration_started: float = 0.0
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    fullscreen: bool = False


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


def positive_seconds(value: str) -> float:
    seconds = float(value.replace(",", "."))
    if not np.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("The delay must be a positive number of seconds.")
    return seconds


def load_settings() -> dict:
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return settings if isinstance(settings, dict) else {}


def save_settings(**values) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps({**load_settings(), **values}), encoding="utf-8")
    except OSError:
        pass


_instance_lock = None


def acquire_instance_lock() -> bool:
    """Only one VanishCam may feed the webcam and the virtual camera at a time."""
    global _instance_lock
    if _instance_lock is not None:
        return True
    try:
        import fcntl
    except ImportError:
        return True
    import tempfile
    handle = open(Path(tempfile.gettempdir()) / "vanishcam.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False
    _instance_lock = handle
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose a disappearing effect for the webcam."
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Webcam index; if omitted, choose the webcam in the start menu",
    )
    parser.add_argument("--width", type=int, default=960, help="Video width")
    parser.add_argument("--height", type=int, default=540, help="Video height")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Start in fullscreen"
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="Do not mirror the preview (the virtual camera is never mirrored)"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="Maximum virtual output FPS (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require the virtual webcam; by default it is enabled when available. --no-virtual-camera disables it",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "v4l2loopback"),
        help="Virtual webcam driver to use",
    )
    parser.add_argument("--virtual-device", help="Name of the virtual device to use")
    parser.add_argument(
        "--mode", choices=ALL_MODES,
        help="Skip the start menu and launch the chosen mode directly",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Print recognition status to the terminal (no text in the video)",
    )
    parser.add_argument(
        "--hud", action=argparse.BooleanOptionalAction, default=True,
        help="Status overlay in the preview window only; never sent to the virtual camera",
    )
    parser.add_argument(
        "--goku-delay", type=positive_seconds, default=None,
        help=f"Goku gesture hold time in seconds (default: {HOLD_SECONDS:g})",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def set_window_topmost(title: str) -> None:
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass


LAUNCH_TITLE = "VanishCam"
_PANEL = (26, 29, 34)
_CARD = (48, 54, 62)
_BORDER = (111, 142, 153)
_ACCENT = (255, 200, 80)
_TEXT = (240, 245, 247)
_MUTED = (179, 195, 201)


def _launch_layout(count: int) -> dict:
    card_step = 82 if count <= 5 else max(48, 410 // count)
    return {
        "cards": [(40, 110 + i * card_step, 420, 110 + i * card_step + card_step - 12)
                  for i in range(count)],
        "preview": (450, 110, 1020, 430),
        "minus": (450, 450, 520, 505),
        "plus": (620, 450, 690, 505),
        "start": (790, 450, 1020, 505),
    }


def _inside(box: tuple[int, int, int, int], x: int, y: int) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _draw_launcher(modes, mode, camera_index, frame, status, layout) -> np.ndarray:
    window = np.full((570, 1060, 3), _PANEL, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(window, "VANISHCAM", (40, 62), font, 1.0, _TEXT, 2, cv2.LINE_AA)
    cv2.putText(window, "Pick an effect and a camera, then press ENTER", (40, 90),
                font, 0.5, _MUTED, 1, cv2.LINE_AA)
    for index, (name, (left, top, right, bottom)) in enumerate(zip(modes, layout["cards"]), 1):
        active = name == mode
        cv2.rectangle(window, (left, top), (right, bottom), (70, 80, 92) if active else _CARD, -1)
        cv2.rectangle(window, (left, top), (right, bottom), _ACCENT if active else _BORDER,
                      3 if active else 1, cv2.LINE_AA)
        title, detail = MODE_LABELS[name]
        cv2.putText(window, f"{index}  {title.upper()}", (left + 18, top + 30),
                    font, 0.7, _TEXT, 2, cv2.LINE_AA)
        cv2.putText(window, detail, (left + 18, top + 55), font, 0.47, _MUTED, 1, cv2.LINE_AA)

    left, top, right, bottom = layout["preview"]
    if frame is not None:
        window[top:bottom, left:right] = cv2.resize(frame, (right - left, bottom - top))
    else:
        cv2.putText(window, status, (left + 120, (top + bottom) // 2), font, 0.65,
                    _MUTED, 2, cv2.LINE_AA)
    cv2.rectangle(window, (left - 1, top - 1), (right, bottom), _BORDER, 1)

    for key, label in (("minus", "-"), ("plus", "+")):
        x0, y0, x1, y1 = layout[key]
        cv2.rectangle(window, (x0, y0), (x1, y1), _CARD, -1)
        cv2.rectangle(window, (x0, y0), (x1, y1), _BORDER, 1)
        cv2.putText(window, label, (x0 + 25, y0 + 38), font, 1.0, _TEXT, 2, cv2.LINE_AA)
    cv2.putText(window, f"CAM {camera_index}", (532, 486), font, 0.6, _TEXT, 2, cv2.LINE_AA)
    x0, y0, x1, y1 = layout["start"]
    ready = frame is not None
    cv2.rectangle(window, (x0, y0), (x1, y1), (78, 132, 124) if ready else _CARD, -1)
    cv2.putText(window, "START  (ENTER)", (x0 + 32, y0 + 36), font, 0.62,
                _TEXT if ready else _MUTED, 2, cv2.LINE_AA)
    cv2.putText(window, f"1-{len(modes)} effect     A / D camera     ENTER start     Q quit",
                (40, 548), font, 0.5, _MUTED, 1, cv2.LINE_AA)
    return window


def choose_launch(modes: tuple[str, ...], mode: str, camera_index: int,
                  mirror: bool = True) -> tuple[str, int] | None:
    """One window: effect cards plus a live preview of the selected webcam."""
    layout = _launch_layout(len(modes))
    selected = {"mode": mode if mode in modes else modes[0],
                "camera": max(0, min(9, camera_index)), "start": False}
    opened_index = None
    capture = None
    frame = None
    read_failures = 0

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONUP:
            return
        for name, box in zip(modes, layout["cards"]):
            if _inside(box, x, y):
                selected["mode"] = name
        if _inside(layout["minus"], x, y):
            selected["camera"] = max(0, selected["camera"] - 1)
        elif _inside(layout["plus"], x, y):
            selected["camera"] = min(9, selected["camera"] + 1)
        elif _inside(layout["start"], x, y):
            selected["start"] = True

    print(f"[Menu] Choose an effect and webcam in the '{LAUNCH_TITLE}' window.", flush=True)
    cv2.namedWindow(LAUNCH_TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(LAUNCH_TITLE, on_mouse)
    raise_pending = True
    try:
        while True:
            if selected["camera"] != opened_index:
                if capture is not None:
                    capture.release()
                try:
                    capture = open_camera(selected["camera"], 640, 360)
                except RuntimeError:
                    capture = None
                opened_index = selected["camera"]
                frame = None
                read_failures = 0
            if capture is not None:
                ok, image = capture.read()
                if ok:
                    frame = cv2.flip(image, 1) if mirror else image
                    read_failures = 0
                else:
                    read_failures += 1
                    if read_failures >= 10:
                        capture.release()
                        capture = None
                        frame = None
            status = ("Waiting for webcam signal..." if capture is not None
                      else "No signal from this webcam")
            cv2.imshow(LAUNCH_TITLE, _draw_launcher(
                modes, selected["mode"], selected["camera"], frame, status, layout))
            if raise_pending:
                set_window_topmost(LAUNCH_TITLE)
                raise_pending = False

            key = cv2.waitKey(30) & 0xFF
            if ord("1") <= key < ord("1") + len(modes):
                selected["mode"] = modes[key - ord("1")]
            elif key in (ord("a"), ord("A")):
                selected["camera"] = max(0, selected["camera"] - 1)
            elif key in (ord("d"), ord("D")):
                selected["camera"] = min(9, selected["camera"] + 1)
            elif key in (10, 13):
                selected["start"] = True
            elif key in (ord("q"), 27):
                return None
            if selected["start"]:
                if frame is not None:
                    return selected["mode"], selected["camera"]
                selected["start"] = False
            if cv2.getWindowProperty(LAUNCH_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyWindow(LAUNCH_TITLE)


def virtual_camera_required(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "virtual_camera", None) is True
                or getattr(args, "virtual_backend", None) or getattr(args, "virtual_device", None))


def open_virtual_camera(args: argparse.Namespace, width: int, height: int, quiet: bool = False):
    if getattr(args, "virtual_camera", None) is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=getattr(args, "virtual_backend", None),
            device=getattr(args, "virtual_device", None),
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        if platform.system() == "Darwin":
            advice = ("Quit the OBS app (its own virtual camera blocks VanishCam). If OBS is not "
                      "installed: brew install --cask obs, then click 'Start Virtual Camera' "
                      "in OBS once to approve the driver.")
        else:
            advice = "Install OBS Studio (OBS Virtual Camera driver) or v4l2loopback on Linux."
        message = f"Virtual webcam unavailable. {advice} Details: {error}"
        if virtual_camera_required(args):
            raise RuntimeError(message) from error
        if not quiet:
            print(f"[Webcam] {message}\n[Webcam] Preview only for now; retrying every few seconds.")
        return None
    print(f"[Webcam] Virtual output active: {camera.device} ({width}x{height}, {args.fps:g} FPS max)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if mode in ("naruto", "goku", "itachi") and destination == GESTURE_MODEL:
            continue
        if mode != "goku" and destination == FACE_MODEL:
            continue
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[VanishCam] Downloading {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "VanishCam/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Cannot download the MediaPipe models. "
                "Check your internet connection and try again."
            ) from error


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} unavailable. Close other apps or try --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def create_tasks(mode: str = "normal") -> tuple[
    vision.ImageSegmenter,
    MultiViewGestureRecognizer | MultiViewHandLandmarker | GokuPoseRecognizer | ItachiPoseRecognizer,
]:
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    if mode == "goku":
        if not HAND_MODEL.is_file() or not FACE_MODEL.is_file():
            raise RuntimeError("Hand or face models missing for Goku.")
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        face_options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        )
        segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
        try:
            hands = vision.HandLandmarker.create_from_options(hand_options)
            try:
                fallback_hands = vision.HandLandmarker.create_from_options(
                    replace(hand_options, running_mode=vision.RunningMode.IMAGE)
                )
                try:
                    face = vision.FaceDetector.create_from_options(face_options)
                except Exception:
                    fallback_hands.close()
                    raise
            except Exception:
                hands.close()
                raise
        except Exception:
            segmenter.close()
            raise
        return segmenter, GokuPoseRecognizer(hands, face, fallback_hands)
    if mode in ("naruto", "itachi"):
        if not HAND_MODEL.is_file():
            raise RuntimeError(f"Hand model not found: {HAND_MODEL}")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        task_class = vision.HandLandmarker
        wrapper_class = MultiViewHandLandmarker
    else:
        classifier_options = mp.tasks.components.processors.ClassifierOptions(
            score_threshold=0.55,
            category_allowlist=["Victory"],
        )
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=classifier_options,
        )
        task_class = vision.GestureRecognizer
        wrapper_class = MultiViewGestureRecognizer

    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        primary = task_class.create_from_options(options)
        try:
            alternate = task_class.create_from_options(
                replace(options, running_mode=vision.RunningMode.IMAGE)
            )
        except Exception:
            primary.close()
            raise
    except Exception:
        segmenter.close()
        raise
    if mode == "itachi":
        return segmenter, ItachiPoseRecognizer(primary, alternate)
    return segmenter, wrapper_class(primary, alternate)


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


LIVE_VANISH_MODES = frozenset(("normal", "particles", "goku"))


def widen_mask(mask: np.ndarray) -> np.ndarray:
    """Solid inside the person, feathered only beyond the outline, so no edge shows."""
    height, width = mask.shape[:2]
    # Work at low resolution: the segmenter's mask is coarse and a large dilation at
    # full HD costs ~100 ms per update.
    small_width = min(width, 320)
    small_size = (small_width, max(1, round(height * small_width / width)))
    solid = (cv2.resize(mask, small_size, interpolation=cv2.INTER_LINEAR) > 0.25).astype(np.float32)
    size = max(5, (small_width // 20) | 1)
    grown = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    feathered = cv2.GaussianBlur(grown, (0, 0), sigmaX=size / 6, sigmaY=size / 6)
    if small_size != (width, height):
        feathered = cv2.resize(feathered, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(feathered, 0.0, 1.0)


def match_exposure(background: np.ndarray, frame: np.ndarray, mask: np.ndarray,
                   previous_gain: np.ndarray | None) -> np.ndarray | None:
    """Per-channel gain that keeps the frozen background in step with auto-exposure."""
    height, width = frame.shape[:2]
    size = (max(1, width // 8), max(1, height // 8))
    outside = cv2.resize(mask, size, interpolation=cv2.INTER_AREA) < 0.05
    if outside.sum() < 50:
        return previous_gain
    live = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)[outside].mean(axis=0)
    base = cv2.resize(background, size, interpolation=cv2.INTER_AREA)[outside].mean(axis=0)
    gain = np.clip(live / np.maximum(base, 1.0), 0.75, 1.33).astype(np.float32)
    return gain if previous_gain is None else previous_gain * 0.7 + gain * 0.3


def apply_gain(image: np.ndarray, gain: np.ndarray | None) -> np.ndarray:
    if gain is None:
        return image
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3 = alpha[:, :, None]
    return (
        person.astype(np.float32) * alpha_3
        + background.astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)


def apply_dissolve(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    progress: float,
    field: np.ndarray,
    sparkle_noise: np.ndarray,
) -> np.ndarray:
    if progress <= 0.002:
        return alpha_composite(frame, background, mask)
    if progress >= 0.999:
        return background.copy()

    points = cv2.findNonZero((mask > 0.01).astype(np.uint8))
    if points is None:
        return background.copy()
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x - 18), max(0, y - 18)
    right = min(mask.shape[1], x + width + 18)
    bottom = min(mask.shape[0], y + height + 18)
    area = np.s_[top:bottom, left:right]
    mask = mask[area]
    field = field[area]
    sparkle_noise = sparkle_noise[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    alpha = mask * survival
    output = alpha_composite(frame[area], background[area], alpha)
    edge = (
        (np.abs(field - eased) < 0.018)
        & (mask > 0.48)
        & (sparkle_noise > 0.54)
    ).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = np.clip(glow * 5.0 + edge, 0.0, 1.0)[:, :, None]
    cyan = np.full_like(output, (255, 228, 92), dtype=np.uint8)
    result = np.clip(
        output.astype(np.float32) * (1.0 - glow * 0.8)
        + cyan.astype(np.float32) * glow * 0.8,
        0,
        255,
    ).astype(np.uint8)
    canvas = background.copy()
    canvas[area] = result
    return canvas


def victory_landmark_score(points: np.ndarray) -> float:
    """Detect extended index and middle fingers using 3D distances, not orientation."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP and tip: index, middle, ring, pinky.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # Seen from the side, folded fingers can look straight but very short:
    # do not require clearly visible joints for the ring and pinky fingers.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # The two tips must be apart; the thumb may stay hidden.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Geometric quality of the extended fingers, not classifier probability.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # Landmarks are returned even when the classifier does not recognize
    # "Victory" (for example with the back of the hand facing the webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # ALWAYS check the visible geometry too: the estimated depth can be
    # inaccurate when the palm and folded fingers are hidden.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


class MultiViewGestureRecognizer:
    """Retry detection on rotated frames without disturbing tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE forces a fresh detection; the VIDEO model stays on the original
        # frame. Avoids inconsistent timestamps/tracking across rotations.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # In the side pose the palm is often too small in the full frame.
        # The two lower crops overlap and cover both hands, even when the
        # image is mirrored.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(rotated)
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class MultiViewHandLandmarker:
    """Track two hands in video and retry difficult views with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(views[crop], rotation)),
            )
            result = self.alternate.detect(rotated)
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        return False

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


def start_vanish_on_gesture(state: AppState, name: str, score: float, now: float) -> bool:
    """Start the dissolve on the first reliable V detection."""
    if state.phase == "ready" and name == "Victory" and score >= 0.62:
        state.phase = "vanishing"
        state.vanish_started = now
        return True
    return False


def start_naruto_on_join(state: AppState, joined: bool, now: float) -> bool:
    if state.phase == "ready" and joined:
        state.phase = "naruto_vanishing"
        state.vanish_started = now
        return True
    return False


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


HUD_LIVE = (90, 210, 90)
HUD_WARN = (40, 170, 255)
HUD_IDLE = (170, 170, 170)


def _hud_text(image, text, origin, scale, color, thickness=1) -> None:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    margin = max(4, height // 3)
    x, y = origin
    _hud_box(image, x - margin, y - height - margin, x + width + margin, y + baseline + margin // 2)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _hud_box(image, x0, y0, x1, y1, color=None) -> None:
    height, width = image.shape[:2]
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return
    if color is None:
        image[y0:y1, x0:x1] = (image[y0:y1, x0:x1] * 0.35).astype(np.uint8)
    else:
        image[y0:y1, x0:x1] = color


HUD_HELP_ITEMS = (
    ("V vanish", "v"), ("R reappear", "r"), ("SPACE new background", " "),
    ("F fullscreen", "f"), ("H hide help", "h"), ("Q quit", "q"),
)


def _hud_button(image, label, x, top, bottom, scale, active=False) -> tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, _), _ = cv2.getTextSize(label, font, 0.55 * scale, 1)
    right = x + text_width + int(18 * scale)
    _hud_box(image, x, top, right, bottom, (80, 200, 255) if active else None)
    baseline = top + (bottom - top) // 2 + int(6 * scale)
    cv2.putText(image, label, (x + int(9 * scale), baseline), font, 0.55 * scale,
                (20, 20, 20) if active else (235, 235, 235), 2 if active else 1, cv2.LINE_AA)
    return x, top, right, bottom


def draw_hud(frame: np.ndarray, mode: str, modes: tuple[str, ...], status: str, detail: str,
             vcam_text: str, vcam_color: tuple[int, int, int], show_help: bool,
             ) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], int]]]:
    """Preview-only overlay plus its click targets; the virtual camera gets the clean frame."""
    image = frame.copy()
    height, width = image.shape[:2]
    scale = max(0.45, width / 1280)
    pad = int(14 * scale)
    gap = int(6 * scale)
    chip_height = int(32 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    targets: list[tuple[tuple[int, int, int, int], int]] = []

    x = pad
    for index, name in enumerate(modes, 1):
        box = _hud_button(image, f"{index} {MODE_LABELS[name][0]}", x, pad, pad + chip_height,
                          scale, active=name == mode)
        targets.append((box, ord(str(index))))
        x = box[2] + gap

    (text_width, _), _ = cv2.getTextSize(vcam_text, font, 0.55 * scale, 1)
    dot = int(7 * scale)
    left = width - pad - text_width - 4 * dot
    _hud_box(image, left, pad, width - pad, pad + chip_height)
    cv2.circle(image, (left + 2 * dot, pad + chip_height // 2), dot, vcam_color, -1, cv2.LINE_AA)
    cv2.putText(image, vcam_text, (left + 3 * dot + 4, pad + int(22 * scale)), font,
                0.55 * scale, (235, 235, 235), 1, cv2.LINE_AA)

    _hud_text(image, status, (pad + int(6 * scale), pad + chip_height + int(44 * scale)),
              0.9 * scale, (255, 255, 255), 2)
    if detail:
        _hud_text(image, detail, (pad + int(6 * scale), pad + chip_height + int(80 * scale)),
                  0.6 * scale, (210, 230, 240))

    bottom = height - pad
    top = bottom - chip_height
    items = HUD_HELP_ITEMS if show_help else (("H help", "h"),)
    x = pad
    for label, key in items:
        if mode == "goku" and key in "vr":
            continue
        box = _hud_button(image, label, x, top, bottom, scale)
        targets.append((box, ord(key)))
        x = box[2] + gap
    return image, targets


def hud_key_at(targets, x: int, y: int) -> int | None:
    for (left, top, right, bottom), key in targets:
        if left <= x <= right and top <= y <= bottom:
            return key
    return None


def hud_status(mode: str, phase: str, calibration_remaining: int | None,
               goku_delay: float, goku_countdown: float | None) -> str:
    if phase == "calibrating":
        seconds = calibration_remaining if calibration_remaining is not None else int(CALIBRATION_SECONDS)
        return f"Step out of the frame: capturing background {seconds}s"
    if phase == "ready":
        if mode == "goku" and goku_countdown is not None:
            return f"Hold it... {goku_countdown:g}s"
        if mode == "goku":
            return f"{GESTURE_HINTS[mode]} for {goku_delay:g}s"
        return GESTURE_HINTS[mode]
    if phase == "vanished":
        if mode == "goku":
            if goku_countdown is not None:
                return f"Reappearing in {goku_countdown:g}s"
            return "Vanished: repeat the gesture to reappear"
        if mode == "itachi":
            return "Vanished: touch your hands at your chest to reappear"
        return "Vanished: press R to reappear"
    if phase in ("goku_appearing", "itachi_appearing"):
        return "Reappearing..."
    return "Vanishing..."


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    for mode in ALL_MODES:
        download_models(mode)
    required = [SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, FACE_MODEL, TELEPORT_SOUND_PATH,
                CROW_SHEET_PATH, SHARINGAN_PATH, LOG_PATH, SOUND_PATH]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing resources: {', '.join(missing)}")
    load_log()
    load_crow_sprites()
    load_sharingan_art()
    for sound_path in (SOUND_PATH, TELEPORT_SOUND_PATH):
        VanishSound(sound_path).close()
    for mode in ALL_MODES:
        segmenter, recognizer = create_tasks(mode)
        try:
            recognizer.close()
        finally:
            segmenter.close()
    print("[Webcam] Self-test passed: models, assets and audio present.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    mode = args.mode
    if mode is None or args.camera is None:
        settings = load_settings()
        remembered_mode = settings.get("mode")
        remembered_camera = settings.get("camera")
        picked = choose_launch(
            ALL_MODES,
            mode or (remembered_mode if remembered_mode in ALL_MODES else ALL_MODES[0]),
            args.camera if args.camera is not None else
            (remembered_camera if isinstance(remembered_camera, int) else 0),
            mirror=not args.no_mirror,
        )
        if picked is None:
            return 0
        mode, args.camera = picked
        save_settings(mode=mode, camera=args.camera)
    with ExitStack() as resources:
        return run_camera(args, resources, mode)


def run_camera(args: argparse.Namespace, resources: ExitStack, mode: str) -> int:
    download_models(mode)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    kits: dict[str, tuple] = {}

    def load_kit(name: str) -> tuple:
        if name not in kits:
            segmenter, recognizer = create_tasks(name)
            resources.callback(segmenter.close)
            resources.callback(recognizer.close)
            sprite = load_log() if name == "naruto" else None
            sound_path = {"naruto": SOUND_PATH, "goku": TELEPORT_SOUND_PATH}.get(name)
            sound = VanishSound(sound_path) if sound_path is not None else None
            if sound is not None:
                resources.callback(sound.close)
            kits[name] = (segmenter, recognizer, sprite, sound)
        return kits[name]

    segmenter, recognizer, log_sprite, vanish_sound = load_kit(mode)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    state = AppState(
        calibration_started=time.perf_counter(), fullscreen=args.fullscreen
    )
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    background_samples: list[np.ndarray] = []
    last_mask: np.ndarray | None = None
    last_mask_at = float("-inf")
    dissolve_field: np.ndarray | None = None
    sparkle_noise: np.ndarray | None = None
    log_anchor: tuple[int, int, int] | None = None
    log_cache: dict[int, np.ndarray] = {}
    vanished_frame: np.ndarray | None = None
    hidden_base_source: np.ndarray | None = None
    hidden_base: np.ndarray | None = None
    hidden_mask: np.ndarray | None = None
    exposure_gain: np.ndarray | None = None
    particle_effect: ParticleEffect | None = None
    itachi_effect: ItachiCrowEffect | None = None
    itachi_gate = ItachiGestureGate() if mode == "itachi" else None
    goku_effect: GokuTeleport | None = None
    goku_delay = getattr(args, "goku_delay", None) or HOLD_SECONDS
    goku_gate = GokuGestureGate(hold_seconds=goku_delay) if mode == "goku" else None
    goku_countdown_remaining: float | None = None
    virtual_camera = None
    virtual_camera_attempts = 0
    virtual_camera_retry_at: float | None = 0.0
    calibration_remaining = None
    last_gesture_report = 0.0
    hud_enabled = getattr(args, "hud", False)
    hud_help = True
    hud_detail = ""
    previous_now = 0.0
    perf_samples: list[float] = []
    perf_last = perf_report_at = time.monotonic()
    pending_keys: list[int] = []
    hud_targets: list[tuple[tuple[int, int, int, int], int]] = []

    def on_preview_click(event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONUP and (key := hud_key_at(hud_targets, x, y)) is not None:
            pending_keys.append(key)

    if hud_enabled:
        cv2.setMouseCallback(WINDOW_NAME, on_preview_click)
    last_timestamp = [-1]
    timestamp_floor = 0

    def stamp() -> int:
        last_timestamp[0] = max(last_timestamp[0] + 1, timestamp_floor)
        return last_timestamp[0]

    started = time.perf_counter()
    print("[Webcam] Step out of the frame for 4 seconds: capturing the background.")
    print("[Webcam] Keys in the preview window: 1-5 effect, SPACE new background, "
          "R reappear, V vanish, F fullscreen, H help, Q quit.")

    raise_pending = True
    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")

            height, width = camera_frame.shape[:2]
            if virtual_camera_retry_at is not None and previous_now >= virtual_camera_retry_at:
                virtual_camera = open_virtual_camera(
                    args, width, height, quiet=virtual_camera_attempts > 0)
                virtual_camera_attempts += 1
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
                    virtual_camera_retry_at = None
                elif getattr(args, "virtual_camera", None) is None:
                    virtual_camera_retry_at = previous_now + 3.0
                else:
                    virtual_camera_retry_at = None
            if mode == "normal" and (dissolve_field is None or dissolve_field.shape != (height, width)):
                dissolve_field, sparkle_noise = make_dissolve_fields(height, width)
            now = time.perf_counter()
            previous_now = now

            timestamp_floor = int((now - started) * 1000)
            mp_image = None

            if (mode == "itachi" and background is not None
                    and state.phase in ("ready", "vanished")
                    and now - state.last_gesture_check >= 0.09):
                state.last_gesture_check = now
                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mask = person_mask(segmenter, mp_image, stamp(), width, height)
                last_mask, last_mask_at = mask, now
                touching = recognizer.recognize_pose(mp_image, stamp(), mask)
                hud_detail = f"Hands seen: {recognizer.last_hand_count}   at chest: {'yes' if touching else 'no'}"
                if args.gesture_debug and now - last_gesture_report >= 1.0:
                    print(f"[Itachi] hands={recognizer.last_hand_count}, chest={touching}")
                    last_gesture_report = now
                if itachi_gate.update(touching, now):
                    try:
                        itachi_effect = ItachiCrowEffect(camera_frame, mask)
                    except ValueError:
                        itachi_gate.reset()
                    else:
                        appearing = state.phase == "vanished"
                        state.phase = "itachi_appearing" if appearing else "itachi_disappearing"
                        state.vanish_started = now
                        vanished_frame = None
                        print("[Itachi] Crows reassembling." if appearing
                              else "[Itachi] Scattering into crows.")

            if (mode == "goku" and background is not None
                    and state.phase in ("ready", "vanished")
                    and now - state.last_gesture_check >= 0.10):
                state.last_gesture_check = now
                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose = recognizer.recognize_pose(mp_image, stamp())
                hud_detail = (f"Face seen: {'yes' if recognizer.last_face_count else 'no'}   "
                              f"hands: {recognizer.last_hand_count}   "
                              f"fingers on forehead: {'yes' if pose else 'no'}")
                if args.gesture_debug and now - last_gesture_report >= 1.0:
                    held = 0.0 if goku_gate.hold_started is None else now - goku_gate.hold_started
                    print(f"[Goku] faces={recognizer.last_face_count}, "
                          f"hands={recognizer.last_hand_count}, pose={pose}, "
                          f"hold={held:.2f}/{goku_gate.hold_seconds:g} s")
                    last_gesture_report = now
                if goku_gate.update(pose, now):
                    action = "Reappearance" if state.phase == "vanished" else "Disappearance"
                    print(f"[Goku] {action}!")
                    goku_countdown_remaining = None
                    mask = person_mask(segmenter, mp_image, stamp(), width, height)
                    goku_effect = GokuTeleport(camera_frame, background, mask)
                    appearing = state.phase == "vanished"
                    state.phase = "goku_appearing" if appearing else "goku_vanishing"
                    state.vanish_started = time.perf_counter()
                    vanish_sound.play()
                elif pose and goku_gate.armed and goku_gate.hold_started is not None:
                    left = goku_gate.hold_seconds - (now - goku_gate.hold_started)
                    remaining = (max(.1,float(np.ceil(left*10)/10)) if goku_gate.hold_seconds < 1
                                 else max(1,int(np.ceil(left))))
                    if remaining != goku_countdown_remaining:
                        action = "Reappearance" if state.phase == "vanished" else "Disappearance"
                        print(f"[Goku] {action} in {remaining:g}...")
                        goku_countdown_remaining = remaining
                elif goku_gate.hold_started is None:
                    goku_countdown_remaining = None

            if state.phase == "calibrating":
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Capturing background: {remaining} seconds.")
                    calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    background_samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    if background_samples:
                        stack = np.stack(background_samples[-18:], axis=0)
                        background = np.median(stack, axis=0).astype(np.uint8)
                    else:
                        background = camera_frame.copy()
                    background_samples.clear()
                    state.phase = "ready"
                    print("[Webcam] Background ready. You can step back into the frame.")
                display = camera_frame

            elif background is None:
                state.phase = "calibrating"
                state.calibration_started = now
                display = camera_frame

            elif mode == "goku" and state.phase in ("goku_vanishing", "goku_appearing"):
                appearing = state.phase == "goku_appearing"
                age = time.perf_counter() - state.vanish_started
                display = goku_effect.render(background, age, appearing)
                if age >= TELEPORT_SECONDS:
                    state.phase = "ready" if appearing else "vanished"
                    vanished_frame = None if appearing else background.copy()

            elif mode == "itachi" and state.phase in ("itachi_disappearing", "itachi_appearing"):
                appearing = state.phase == "itachi_appearing"
                age = max(0.0, time.perf_counter() - state.vanish_started)
                display = itachi_effect.render(background, age, appearing)
                if age >= ITACHI_SECONDS:
                    state.phase = "ready" if appearing else "vanished"
                    vanished_frame = None if appearing else display
                    itachi_effect = None

            elif state.phase == "vanished":
                if vanished_frame is None:
                    vanished_frame = background.copy()
                    if mode == "naruto" and log_anchor is not None:
                        overlay_log(vanished_frame, log_sprite, landed_log_anchor(log_anchor, height),
                                    cache=log_cache)
                if mode in LIVE_VANISH_MODES:
                    if hidden_base_source is not vanished_frame:
                        hidden_base_source = hidden_base = vanished_frame
                        hidden_mask = exposure_gain = None
                    if hidden_mask is None or now - last_mask_at >= 0.05:
                        if mp_image is None:
                            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        fresh = widen_mask(person_mask(segmenter, mp_image, stamp(), width, height))
                        # Hold the previous mask briefly so a missed frame cannot flash a limb.
                        hidden_mask = fresh if hidden_mask is None else np.maximum(fresh, hidden_mask * 0.8)
                        last_mask = hidden_mask
                        last_mask_at = now
                        exposure_gain = match_exposure(vanished_frame, camera_frame, hidden_mask, exposure_gain)
                        hidden_base = apply_gain(vanished_frame, exposure_gain)
                    display = alpha_composite(hidden_base, camera_frame, hidden_mask)
                else:
                    display = vanished_frame

            else:
                mask = last_mask
                if (
                    state.phase == "ready"
                    and mode not in ("goku", "itachi")
                    and now - state.last_gesture_check >=
                    (0.08 if mode == "naruto" else GESTURE_INTERVAL_SECONDS)
                ):
                    state.last_gesture_check = now
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if mode != "naruto":
                        name, score = recognize_gesture(recognizer, mp_image, stamp())
                        hud_detail = ("No hand in view" if name == "No_Hand"
                                      else f"Seen: {name} ({score:.0%}, needs V sign over 62%)")
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(
                                f"[Gesture] {name}, score={score:.2f}, "
                                f"crop={recognizer.preferred_crop or 'full'}, "
                                f"rotation={recognizer.preferred_rotation or 0}"
                            )
                            last_gesture_report = now
                        start_vanish_on_gesture(state, name, score, now)
                    else:
                        joined = recognizer.recognize_joined_hands(mp_image, stamp())
                        hud_detail = (f"Hands seen: {recognizer.last_hand_count}   "
                                      f"palms together: {'yes' if joined else 'no'}")
                        if args.gesture_debug and now - last_gesture_report >= 1.0:
                            print(f"[Hands] detected={recognizer.last_hand_count}, "
                                  f"joined={joined}, view={recognizer.preferred_view or 'full'}")
                            last_gesture_report = now
                        if start_naruto_on_join(state, joined, time.perf_counter()):
                            vanish_sound.play()

                if state.phase in ("vanishing", "naruto_vanishing"):
                    # Normal tracks the silhouette at 20 Hz. Particles captures
                    # it once; Naruto needs it only while the person is visible.
                    needs_mask = (
                        (mode == "normal" and (mask is None or now - last_mask_at >= 0.05))
                        or (mode == "particles" and particle_effect is None)
                        or (mode == "naruto" and (log_anchor is None or now - state.vanish_started < 0.13))
                    )
                    if needs_mask:
                        if mp_image is None:
                            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        mask = person_mask(segmenter, mp_image, stamp(), width, height)
                        last_mask = mask
                        last_mask_at = now
                    if mode == "naruto" and log_anchor is None:
                        log_anchor = substitution_anchor(mask)

                if mode == "naruto" and state.phase == "naruto_vanishing":
                    age = max(0.04, time.perf_counter() - state.vanish_started)
                    display = render_substitution(
                        camera_frame, background, mask, log_sprite, log_anchor, age, log_cache
                    )
                    if age >= EFFECT_SECONDS:
                        state.phase = "vanished"
                        vanished_frame = display
                elif mode == "particles" and state.phase == "vanishing":
                    if particle_effect is None:
                        particle_effect = ParticleEffect(camera_frame, mask)
                    age = time.perf_counter() - state.vanish_started
                    display = particle_effect.render(background, age)
                    if age >= PARTICLE_SECONDS:
                        state.phase = "vanished"
                        vanished_frame = display
                elif state.phase == "ready":
                    display = camera_frame
                else:
                    progress = 0.0
                    if state.phase == "vanishing" and state.vanish_started is not None:
                        progress = min(
                            1.0, (time.perf_counter() - state.vanish_started) / VANISH_SECONDS
                        )
                        if progress >= 1.0:
                            state.phase = "vanished"
                    display = apply_dissolve(
                        camera_frame, background, mask, progress, dissolve_field, sparkle_noise
                    )
                    if state.phase == "vanished":
                        vanished_frame = display

            if virtual_camera is not None:
                send_virtual_frame(virtual_camera, display)
            # Mirror the preview only: video apps expect the camera's true orientation
            # and mirror your self-view themselves.
            preview = display if args.no_mirror else cv2.flip(display, 1)
            if hud_enabled:
                if virtual_camera is not None:
                    vcam_text, vcam_color = f"Live: {virtual_camera.device}", HUD_LIVE
                elif virtual_camera_retry_at is not None:
                    vcam_text, vcam_color = "Virtual camera busy (quit OBS?)", HUD_WARN
                else:
                    vcam_text, vcam_color = "Preview only", HUD_IDLE
                preview, hud_targets[:] = draw_hud(
                    preview, mode, ALL_MODES,
                    hud_status(mode, state.phase, calibration_remaining, goku_delay,
                               goku_countdown_remaining),
                    hud_detail if state.phase in ("ready", "vanished") else "",
                    vcam_text, vcam_color, hud_help,
                )
            cv2.imshow(WINDOW_NAME, preview)
            if raise_pending:
                set_window_topmost(WINDOW_NAME)
                raise_pending = False

            key = cv2.waitKey(1) & 0xFF
            if pending_keys:
                key = pending_keys.pop(0)
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if ord("1") <= key < ord("1") + len(ALL_MODES) and ALL_MODES[key - ord("1")] != mode:
                mode = ALL_MODES[key - ord("1")]
                print(f"[Webcam] Effect: {MODE_LABELS[mode][0]}")
                download_models(mode)
                segmenter, recognizer, log_sprite, vanish_sound = load_kit(mode)
                itachi_gate = ItachiGestureGate() if mode == "itachi" else None
                goku_gate = GokuGestureGate(hold_seconds=goku_delay) if mode == "goku" else None
                if background is not None:
                    state.phase = "ready"
                state.vanish_started = None
                log_anchor = None
                last_mask = None
                last_mask_at = float("-inf")
                vanished_frame = None
                particle_effect = None
                itachi_effect = None
                goku_effect = None
                goku_countdown_remaining = None
                hud_detail = ""
            elif key in (ord("h"), ord("H")):
                hud_help = not hud_help
            elif key == ord(" "):
                state.phase = "calibrating"
                state.calibration_started = now
                state.vanish_started = None
                log_anchor = None
                last_mask = None
                last_mask_at = float("-inf")
                vanished_frame = None
                particle_effect = None
                itachi_effect = None
                if itachi_gate is not None:
                    itachi_gate.reset()
                goku_effect = None
                if goku_gate is not None:
                    goku_gate.reset()
                    goku_countdown_remaining = None
                background_samples.clear()
                calibration_remaining = None
                print("[Webcam] Step out of the frame for 4 seconds: capturing the new background.")
            elif key == ord("r"):
                if background is not None and mode == "itachi" and state.phase == "vanished":
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    mask = person_mask(segmenter, mp_image, stamp(), width, height)
                    try:
                        itachi_effect = ItachiCrowEffect(camera_frame, mask)
                    except ValueError:
                        print("[Itachi] Step back into the frame before reappearing.")
                    else:
                        state.phase = "itachi_appearing"
                        state.vanish_started = time.perf_counter()
                        vanished_frame = None
                elif background is not None and mode not in ("goku", "itachi"):
                    state.phase = "ready"
                    state.vanish_started = None
                    vanished_frame = None
                    log_anchor = None
                    last_mask = None
                    last_mask_at = float("-inf")
                    particle_effect = None
            elif key == ord("v") and background is not None and mode == "itachi" and state.phase == "ready":
                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mask = person_mask(segmenter, mp_image, stamp(), width, height)
                try:
                    itachi_effect = ItachiCrowEffect(camera_frame, mask)
                except ValueError:
                    print("[Itachi] No person detected.")
                else:
                    state.phase = "itachi_disappearing"
                    state.vanish_started = time.perf_counter()
            elif key == ord("v") and background is not None and mode not in ("goku", "itachi"):
                state.phase = "naruto_vanishing" if mode == "naruto" else "vanishing"
                state.vanish_started = time.perf_counter()
                if mode == "naruto":
                    vanish_sound.play()
                    log_anchor = None
                vanished_frame = None
                particle_effect = None
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            if args.gesture_debug:
                loop_end = time.monotonic()
                perf_samples.append(loop_end - perf_last)
                perf_last = loop_end
                if loop_end - perf_report_at >= 2.0:
                    average = sum(perf_samples) / len(perf_samples)
                    print(f"[Perf] {1 / average:.1f} fps, frame {average * 1000:.0f} ms avg, "
                          f"{max(perf_samples) * 1000:.0f} ms worst")
                    perf_samples.clear()
                    perf_report_at = loop_end
    finally:
        resources.close()

    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    try:
        if any(flag in sys.argv[1:] for flag in ("--self-test", "--help", "-h")):
            return run()  # Model/resource checks only; never opens a camera.
        if not acquire_instance_lock():
            print("[VanishCam] Already running. Quit the other window (Q) first.", file=sys.stderr)
            return 1
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[VanishCam] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
