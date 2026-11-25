"""Finger-snap vanish: a puff of smoke hides you, another brings you back."""

from __future__ import annotations

import cv2
import numpy as np

import mediapipe as mp

from naruto_effect import EFFECT_SECONDS as SMOKE_SECONDS, draw_smoke

SNAP_SECONDS = SMOKE_SECONDS
HIDE_SECONDS = 0.13       # You disappear inside the first burst of smoke.
REVEAL_START = 0.25       # When reappearing, the smoke blooms before you fade in...
REVEAL_SECONDS = 0.3      # ...over this long.

# Calibrated on labelled recordings at 30 fps (snaps vs a held pinch, OK sign, waving, fists).
# A snap lasts one or two frames, so the camera sees its two ends: thumb pressed on the
# middle fingertip with the finger extended, then the fingertip curled down into the palm.
# Distances are in hand lengths (wrist to middle knuckle).
PINCH_MAX = 0.15           # Thumb tip to middle tip while pressed.
PINCH_EXTENSION_MIN = 0.95 # Middle finger reaching past its knuckle.
FLICK_OPEN_MIN = 0.5       # Thumb and middle tip apart after the flick...
FLICK_CURL_MAX = 0.75      # ...with the middle tip down at the palm.
SNAP_WINDOW = 0.25         # Pinch to flick; a held pinch released slowly does not count.
SNAP_COOLDOWN = 0.8
WRIST, MIDDLE_KNUCKLE, THUMB_TIP, MIDDLE_TIP = 0, 9, 4, 12


def snap_measures(hand, width: int, height: int) -> tuple[float, float]:
    """(thumb-middle gap, middle extension) for one hand's 21 landmarks."""
    points = np.array([(point.x * width, point.y * height) for point in hand], np.float32)
    length = max(float(np.linalg.norm(points[MIDDLE_KNUCKLE] - points[WRIST])), 1e-6)
    gap = float(np.linalg.norm(points[THUMB_TIP] - points[MIDDLE_TIP])) / length
    extension = float(np.linalg.norm(points[MIDDLE_TIP] - points[WRIST])) / length
    return gap, extension


class SnapDetector:
    """Fires once per pinch-then-flick of any hand."""

    def __init__(self):
        self.last_pinch: float | None = None
        self.last_fire = float("-inf")
        self.pinched = False

    def update(self, hands, width: int, height: int, now: float) -> bool:
        self.pinched = False
        flicked = False
        for hand in hands:
            gap, extension = snap_measures(hand, width, height)
            if gap < PINCH_MAX and extension > PINCH_EXTENSION_MIN:
                self.pinched = True
            elif gap > FLICK_OPEN_MIN and extension < FLICK_CURL_MAX:
                flicked = True
        if self.pinched:
            self.last_pinch = now
            return False
        if (flicked and self.last_pinch is not None and now - self.last_pinch <= SNAP_WINDOW
                and now - self.last_fire >= SNAP_COOLDOWN):
            self.last_fire = now
            self.last_pinch = None
            return True
        return False

    def reset(self) -> None:
        self.last_pinch = None
        self.pinched = False


class SnapRecognizer:
    """Hand landmarks on every frame, fed to a SnapDetector."""

    def __init__(self, landmarker):
        self.landmarker = landmarker
        self.detector = SnapDetector()
        self.last_hand_count = 0

    def recognize_snap(self, image: mp.Image, timestamp_ms: int, now: float) -> bool:
        hands = self.landmarker.detect_for_video(image, timestamp_ms).hand_landmarks
        return self.observe(hands, image.width, image.height, now)

    def observe(self, hands, width: int, height: int, now: float) -> bool:
        """Check hand landmarks someone else already detected on this frame."""
        self.last_hand_count = len(hands)
        return self.detector.update(hands, width, height, now)

    def describe(self) -> str:
        if not self.last_hand_count:
            return "No hand in view"
        return f"Hand seen   thumb on middle finger: {'yes' if self.detector.pinched else 'no'}"

    def close(self) -> None:
        self.landmarker.close()


def snap_anchor(cover: np.ndarray) -> tuple[int, int, int]:
    """Centre and size the smoke so the cloud covers the whole person, not just the torso."""
    height, width = cover.shape[:2]
    points = cv2.findNonZero((cover > 0.5).astype(np.uint8))
    if points is None:
        return width // 2, height // 2, int(height * 0.6)
    x, y, w, h = cv2.boundingRect(points)
    size = int(np.clip(max(h, w * 1.3) * 0.75, height * 0.3, height * 0.95))
    return x + w // 2, y + int(h * 0.55), size


def person_visibility(age: float, appearing: bool) -> float:
    if appearing:
        return float(np.clip((age - REVEAL_START) / REVEAL_SECONDS, 0.0, 1.0))
    return float(np.clip(1.0 - age / HIDE_SECONDS, 0.0, 1.0))


def render_snap(scene: np.ndarray, plate: np.ndarray, cover: np.ndarray,
                anchor: tuple[int, int, int], age: float, appearing: bool) -> np.ndarray:
    """The live scene with the person (cover mask) hidden or revealed under a smoke puff.

    Only the person's area switches to the background plate, so the room stays live.
    """
    visible = person_visibility(age, appearing)
    alpha = (1.0 - cover * (1.0 - visible))[:, :, None]
    output = (scene.astype(np.float32) * alpha
              + plate.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    # A snap is instant: start with a full cloud instead of Naruto's growing wisp.
    draw_smoke(output, anchor, age, foreground=False, burst=0.7, growth=0.3)
    draw_smoke(output, anchor, age, foreground=True, burst=0.6, growth=0.3)
    return output
