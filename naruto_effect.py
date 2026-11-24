"""Two-hand trigger and smoke/log compositing for body substitution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


EFFECT_SECONDS = 1.45
LOG_PATH = Path(__file__).resolve().parent / "assets" / "substitution_log.png"


FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
WRIST, MIDDLE_KNUCKLE = 0, 9
PALM = (0, 5, 9, 13, 17)

# Calibrated on labelled recordings (seal at chest and face vs one open hand, karate chop,
# waving, hands apart, hands resting / on a knee). Distances are relative to hand length.
# MediaPipe sees pressed palms either as two hands almost on top of each other, or merges
# them into one hand that is narrow (seen side-on) and distorted. A real single hand is
# never distorted that way, which is what keeps a raised hand or a wave from triggering.
TILT_MAX = 30.0           # Degrees from straight up in the image.
PAIR_PALMS_MAX = 0.7      # Two hands: distance between palm centres.
MERGED_WIDTH_MAX = 0.45   # One hand: knuckle width (index to pinky), side-on is narrow.
MERGED_GAP_MIN = 0.7      # One hand: widest neighbouring fingertip gap, distorted if large...
MERGED_STRAIGHT_MAX = 0.85  # ...or least-straight finger chord/path, distorted if bent.
MERGED_LOWEST = 0.85      # One hand: palm centre must be above the bottom 15% of the frame.


@dataclass(frozen=True)
class HandSeal:
    """Result of the Naruto seal check, with the parts that are missing."""

    hands: int
    in_frame: bool = False
    pointing_up: bool = False
    palms_together: bool = False
    measures: dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def ok(self) -> bool:
        return self.hands >= 1 and self.in_frame and self.pointing_up and self.palms_together

    def describe(self) -> str:
        if self.hands == 0:
            return "No hands in view"
        if not self.in_frame:
            return "Hands cut off by the frame edge"
        mark = {True: "yes", False: "no"}
        return (f"Hands seen: {self.hands}   pointing up: {mark[self.pointing_up]}   "
                f"palms pressed together: {mark[self.palms_together]}")


def _hand_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[MIDDLE_KNUCKLE] - points[WRIST]))


def _straightness(points: np.ndarray) -> float:
    """Least straight finger: knuckle-to-tip chord / path along the finger (1.0 = straight)."""
    ratios = []
    for joints in FINGERS:
        segments = sum(float(np.linalg.norm(points[b] - points[a]))
                       for a, b in zip(joints, joints[1:]))
        chord = float(np.linalg.norm(points[joints[-1]] - points[joints[0]]))
        ratios.append(chord / segments if segments > 0 else 0.0)
    return min(ratios)


def _finger_gap(points: np.ndarray) -> float:
    """Widest gap between neighbouring fingertips, relative to hand length."""
    tips = [points[joints[-1]] for joints in FINGERS]
    widest = max(float(np.linalg.norm(b - a)) for a, b in zip(tips, tips[1:]))
    return widest / max(_hand_length(points), 1e-6)


def _tilt(points: np.ndarray) -> float:
    direction = points[MIDDLE_KNUCKLE] - points[WRIST]
    # Image y grows downwards, so "up" is negative y.
    return math.degrees(math.atan2(abs(direction[0]), -direction[1]))


def analyze_hand_seal(hands, width: int, height: int) -> HandSeal:
    """Check the Naruto seal (palms pressed together, pointing up) in one frame."""
    hands = [hand for hand in hands[:2] if len(hand) == 21]
    if not hands:
        return HandSeal(hands=0)
    # Hands cut off by the frame edge (e.g. resting on a knee below the camera) are guesses
    # by the landmarker, not a deliberate pose.
    if any(not (0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0) for hand in hands for p in hand):
        return HandSeal(hands=len(hands))
    points = [np.array([(p.x * width, p.y * height) for p in hand]) for hand in hands]
    if not all(np.isfinite(hand).all() for hand in points):
        return HandSeal(hands=len(hands))
    lengths = [max(_hand_length(hand), 1e-6) for hand in points]
    tilt = max(_tilt(hand) for hand in points)

    if len(points) == 2:
        palms = [hand[list(PALM)].mean(axis=0) for hand in points]
        distance = float(np.linalg.norm(palms[0] - palms[1])) / (sum(lengths) / 2)
        return HandSeal(hands=2, in_frame=True, pointing_up=tilt < TILT_MAX,
                        palms_together=distance < PAIR_PALMS_MAX,
                        measures={"tilt": tilt, "palms": distance})

    hand = points[0]
    measures = {
        "tilt": tilt,
        "width": float(np.linalg.norm(hand[5] - hand[17])) / lengths[0],
        "gap": _finger_gap(hand),
        "straight": _straightness(hand),
        "height": float(hand[list(PALM)].mean(axis=0)[1]) / height,
    }
    merged = (measures["width"] < MERGED_WIDTH_MAX and measures["height"] < MERGED_LOWEST
              and (measures["gap"] > MERGED_GAP_MIN or measures["straight"] < MERGED_STRAIGHT_MAX))
    return HandSeal(hands=1, in_frame=True, pointing_up=tilt < TILT_MAX,
                    palms_together=merged, measures=measures)


def joined_hands_from_landmarks(hands, width: int, height: int) -> bool:
    return analyze_hand_seal(hands, width, height).ok


def substitution_anchor(mask: np.ndarray) -> tuple[int, int, int]:
    """Place the log at the person's torso instead of at a fixed screen spot."""
    height, width = mask.shape
    binary = (mask > 0.45).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return width // 2, height // 2, int(height * 0.57)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[largest]
    size = int(np.clip(h * 0.58, height * 0.24, height * 0.70))
    center_x = int(np.clip(x + w * 0.5, size // 3, width - size // 3))
    center_y = int(np.clip(y + h * 0.52, size // 3, height - size // 3))
    return center_x, center_y, size


def load_log() -> np.ndarray:
    sprite = cv2.imread(str(LOG_PATH), cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
        raise RuntimeError(f"Log sprite unavailable: {LOG_PATH}")
    return sprite


def overlay_log(frame: np.ndarray, sprite: np.ndarray, anchor: tuple[int, int, int],
                opacity: float = 1.0, scale: float = 1.0,
                cache: dict[int, np.ndarray] | None = None) -> None:
    if opacity <= 0:
        return
    cx, cy, target_height = anchor
    height = max(1, int(target_height * scale))
    width = max(1, int(height * sprite.shape[1] / sprite.shape[0]))
    sized = cache.get(height) if cache is not None else None
    if sized is None:
        sized = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
        if cache is not None:
            cache[height] = sized
    x1, y1 = cx - width // 2, cy - height // 2
    x2, y2 = x1 + width, y1 + height
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if left >= right or top >= bottom:
        return
    part = sized[top - y1:bottom - y1, left - x1:right - x1]
    alpha = (part[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    destination = frame[top:bottom, left:right]
    destination[:] = np.clip(
        part[:, :, :3].astype(np.float32) * alpha
        + destination.astype(np.float32) * (1.0 - alpha),
        0, 255,
    ).astype(np.uint8)


def landed_log_anchor(anchor: tuple[int, int, int], frame_height: int) -> tuple[int, int, int]:
    cx, cy, size = anchor
    return cx, max(cy, frame_height - round(size * 0.48)), size


def falling_log_anchor(anchor: tuple[int, int, int], frame_height: int,
                       age: float) -> tuple[int, int, int]:
    """Accelerate downwards, touch the frame floor, then settle with a small bounce."""
    cx, cy, size = anchor
    floor_y = landed_log_anchor(anchor, frame_height)[1]
    distance = floor_y - cy
    if distance <= 0 or age <= 0.18:
        return anchor
    elapsed = age - 0.18
    gravity = frame_height * 3.2
    fall = gravity * elapsed * elapsed * 0.5
    if fall < distance:
        return cx, cy + round(fall), size
    landed_at = math.sqrt(2.0 * distance / gravity)
    bounce_age = elapsed - landed_at
    bounce = size * 0.075 * math.exp(-5.0 * bounce_age) * abs(math.sin(11.0 * bounce_age))
    return cx, floor_y - round(bounce), size


def draw_smoke(frame: np.ndarray, anchor: tuple[int, int, int], age: float,
               foreground: bool, burst: float = 0.27, growth: float = 0.54) -> None:
    """Expanding smoke puffs around anchor; burst is the starting radius (share of size)."""
    if age >= EFFECT_SECONDS:
        return
    cx, cy, size = anchor
    rng = np.random.default_rng(9127 if foreground else 1934)
    reveal = min(1.0, age / 0.16)
    fade = max(0.0, 1.0 - max(0.0, age - 0.33) / 1.12)
    if fade <= 0:
        return
    # Smoke only occupies the area around the log; avoid allocating and
    # blending a full-size RGBA video frame twice per animation frame.
    margin = int(size * 1.35)
    left, top = max(0, cx - margin), max(0, cy - margin)
    right = min(frame.shape[1], cx + margin + 1)
    bottom = min(frame.shape[0], cy + margin + 1)
    if left >= right or top >= bottom:
        return
    layer = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
    radius = size * (burst + growth * min(age, 1.0))
    for _ in range(22 if foreground else 18):
        angle = rng.uniform(0, 2 * math.pi)
        distance = radius * rng.uniform(0.25, 1.0)
        x = int(cx + math.cos(angle) * distance) - left
        y = int(cy + math.sin(angle) * distance * 0.63 - age * size * 0.13) - top
        puff = max(2, int(size * rng.uniform(0.09, 0.20) * (0.7 + age * 0.7)))
        color = (229, 235, 235, 190) if foreground else (152, 162, 166, 125)
        cv2.ellipse(layer, (x, y), (puff, int(puff * 0.68)),
                    rng.uniform(-25, 25), 0, 360, color, -1, cv2.LINE_AA)
    # Long white winding smoke ribbons echo the supplied anime reference.
    for direction in (-1, 1):
        center = (int(cx + direction * size * 0.13) - left,
                  int(cy - size * 0.13 - age * size * 0.09) - top)
        axes = (max(2, int(size * (0.30 + age * 0.27))),
                max(2, int(size * (0.09 + age * 0.12))))
        cv2.ellipse(layer, center, axes, direction * 15, 20, 230,
                    (244, 247, 245, 220), max(2, int(size * 0.025)), cv2.LINE_AA)
    occupied = cv2.findNonZero(layer[:, :, 3])
    if occupied is None:
        return
    x, y, width, height = cv2.boundingRect(occupied)
    destination = frame[top + y:top + y + height, left + x:left + x + width]
    layer = layer[y:y + height, x:x + width]
    alpha = (layer[:, :, 3:4].astype(np.float32) / 255.0) * reveal * fade
    destination[:] = np.clip(
        layer[:, :, :3].astype(np.float32) * alpha
        + destination.astype(np.float32) * (1.0 - alpha),
        0, 255,
    ).astype(np.uint8)


def render_substitution(frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
                        sprite: np.ndarray, anchor: tuple[int, int, int], age: float,
                        log_cache: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Quick body replacement, expanding smoke, then a persistent wooden log."""
    if age >= EFFECT_SECONDS:
        output = background.copy()
        overlay_log(output, sprite, landed_log_anchor(anchor, output.shape[0]), cache=log_cache)
        return output
    if age >= 0.13:
        output = background.copy()
    else:
        person_alpha = mask * max(0.0, 1.0 - age / 0.13)
        output = (
            frame.astype(np.float32) * person_alpha[:, :, None]
            + background.astype(np.float32) * (1.0 - person_alpha[:, :, None])
        ).astype(np.uint8)
    draw_smoke(output, anchor, age, foreground=False)
    log_opacity = min(1.0, max(0.0, (age - 0.05) / 0.15))
    moving_anchor = falling_log_anchor(anchor, output.shape[0], age)
    overlay_log(output, sprite, moving_anchor, log_opacity,
                scale=1.15 - min(age, 0.25) * 0.6, cache=log_cache)
    draw_smoke(output, anchor, age, foreground=True)
    return output
