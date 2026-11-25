"""Shadow clones: live copies of you pop out of smoke beside you and vanish the same way."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

from naruto_effect import EFFECT_SECONDS as SMOKE_SECONDS, draw_smoke

# (horizontal offset as a share of frame width, scale). The far pair is smaller and drawn
# first, so it reads as standing behind the near pair.
CLONE_SLOTS = ((-0.44, 0.74), (0.44, 0.74), (-0.26, 0.88), (0.26, 0.88))
POP_ORDER = (2, 3, 0, 1)          # Near clones pop first.
POP_STAGGER = 0.14
POP_HIDE_SECONDS = 0.12           # A clone appears (or goes) inside its puff's first burst.
POP_REVEAL_START = 0.1
CLONE_SECONDS = POP_STAGGER * (len(CLONE_SLOTS) - 1) + SMOKE_SECONDS


# Calibrated on labelled recordings: the seal at chest and face height vs palms together,
# V signs with both hands, touching or interlocked hands, one hand, resting hands.
# Extensions are fingertip-to-wrist over knuckle-to-wrist; distances in hand lengths.
SEAL_FINGERS_MIN = 1.4      # Index and middle straight (seal ~1.8, curled < 1).
SEAL_FOLDED_MAX = 1.35      # Ring and pinky folded (seal ~1.15, open hands > 1.6).
SEAL_ANGLE_MIN = 0.75       # |sin| between the two hands' finger lines (seal ~0.95).
SEAL_GAP_MAX = 0.55         # Finger midpoints close together (seal ~0.35, apart > 1).
WRIST, INDEX_KNUCKLE, INDEX_TIP, MIDDLE_KNUCKLE, MIDDLE_TIP = 0, 5, 8, 9, 12
RING_KNUCKLE, RING_TIP, PINKY_KNUCKLE, PINKY_TIP = 13, 16, 17, 20


@dataclass(frozen=True)
class CloneSeal:
    """Result of the clone seal check: index and middle fingers of both hands crossed."""

    hands: int
    fingers: bool = False
    crossed: bool = False
    measures: dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def ok(self) -> bool:
        return self.hands == 2 and self.fingers and self.crossed

    def describe(self) -> str:
        if self.hands < 2:
            return "Show both hands" if self.hands else "No hands in view"
        mark = {True: "yes", False: "no"}
        return f"Two fingers out: {mark[self.fingers]}   crossed: {mark[self.crossed]}"


def _extension(points: np.ndarray, tip: int, knuckle: int) -> float:
    return float(np.linalg.norm(points[tip] - points[WRIST])
                 / max(np.linalg.norm(points[knuckle] - points[WRIST]), 1e-6))


def _segments_cross(a0, a1, b0, b1) -> bool:
    def side(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return side(a0, a1, b0) != side(a0, a1, b1) and side(b0, b1, a0) != side(b0, b1, a1)


def analyze_clone_seal(hands, width: int, height: int) -> CloneSeal:
    if len(hands) != 2:
        return CloneSeal(hands=len(hands))
    points = [np.array([(p.x * width, p.y * height) for p in hand], np.float64) for hand in hands]
    fingers_out = min(min(_extension(p, INDEX_TIP, INDEX_KNUCKLE), _extension(p, MIDDLE_TIP, MIDDLE_KNUCKLE))
                      for p in points)
    folded = max(max(_extension(p, RING_TIP, RING_KNUCKLE), _extension(p, PINKY_TIP, PINKY_KNUCKLE))
                 for p in points)
    lines, directions = [], []
    for p in points:
        base = (p[INDEX_KNUCKLE] + p[MIDDLE_KNUCKLE]) / 2
        tip = (p[INDEX_TIP] + p[MIDDLE_TIP]) / 2
        along = tip - base
        directions.append(along / max(np.linalg.norm(along), 1e-6))
        lines.append((base - along * 0.3, tip + along * 0.3))  # Slack for fingertip jitter.
    angle = abs(float(directions[0][0] * directions[1][1] - directions[0][1] * directions[1][0]))
    hand_length = float(np.mean([np.linalg.norm(p[MIDDLE_KNUCKLE] - p[WRIST]) for p in points]))
    gap = float(np.linalg.norm(sum(lines[0]) / 2 - sum(lines[1]) / 2) / max(hand_length, 1e-6))
    return CloneSeal(
        hands=2,
        fingers=fingers_out > SEAL_FINGERS_MIN and folded < SEAL_FOLDED_MAX,
        crossed=angle > SEAL_ANGLE_MIN and gap < SEAL_GAP_MAX and _segments_cross(*lines[0], *lines[1]),
        measures={"fingers": fingers_out, "folded": folded, "angle": angle, "gap": gap},
    )


class CloneSealRecognizer:
    """Hand landmarks fed to analyze_clone_seal."""

    def __init__(self, landmarker):
        self.landmarker = landmarker
        self.last_seal = CloneSeal(hands=0)

    def recognize_seal(self, image: mp.Image, timestamp_ms: int) -> bool:
        hands = self.landmarker.detect_for_video(image, timestamp_ms).hand_landmarks
        self.last_seal = analyze_clone_seal(hands, image.width, image.height)
        return self.last_seal.ok

    def close(self) -> None:
        self.landmarker.close()


def _placement(anchor_x: float, offset: float, scale: float, width: int, height: int) -> np.ndarray:
    """Scale about the person's feet (frame bottom) and shift sideways."""
    shift = anchor_x + offset * width - scale * anchor_x
    return np.float32([[scale, 0, shift], [0, scale, height - scale * height]])


def person_center_x(mask: np.ndarray) -> float | None:
    weight = mask.sum(axis=0)
    total = float(weight.sum())
    if total < mask.shape[0] * 2:
        return None
    return float((weight * np.arange(mask.shape[1])).sum() / total)


class CloneSquad:
    """Renders the clones for one appearance; anchored where the person stood at the seal."""

    def __init__(self, mask: np.ndarray):
        height, width = mask.shape[:2]
        center = person_center_x(mask)
        if center is None:
            raise ValueError("No person to clone.")
        self.size = (width, height)
        self.matrices = [_placement(center, offset, scale, width, height)
                         for offset, scale in CLONE_SLOTS]
        self.puffs = []
        for matrix in self.matrices:
            placed = cv2.warpAffine(mask, matrix, (width, height))
            points = cv2.findNonZero((placed > 0.5).astype(np.uint8))
            if points is None:
                self.puffs.append(None)
                continue
            x, y, w, h = cv2.boundingRect(points)
            size = int(np.clip(max(h, w * 1.3) * 0.7, height * 0.25, height * 0.9))
            self.puffs.append((x + w // 2, y + int(h * 0.5), size))

    def _visibility(self, slot: int, age: float, phase: str) -> float:
        if phase == "steady":
            return 1.0
        local = age - POP_ORDER.index(slot) * POP_STAGGER
        if phase == "appearing":
            return float(np.clip((local - POP_REVEAL_START) / 0.15, 0.0, 1.0))
        return float(np.clip(1.0 - local / POP_HIDE_SECONDS, 0.0, 1.0))

    def render(self, scene: np.ndarray, mask: np.ndarray, age: float = 0.0,
               phase: str = "steady") -> np.ndarray:
        """phase: "appearing", "steady" or "leaving"; mask is the live person mask."""
        width, height = self.size
        output = scene.astype(np.float32)
        for slot, matrix in enumerate(self.matrices):
            visible = self._visibility(slot, age, phase)
            if visible <= 0:
                continue
            clone = cv2.warpAffine(scene, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
            alpha = cv2.warpAffine(mask, matrix, (width, height))[:, :, None] * visible
            output = output * (1 - alpha) + clone.astype(np.float32) * alpha
        live = mask[:, :, None]
        output = (output * (1 - live) + scene.astype(np.float32) * live).astype(np.uint8)
        if phase != "steady":
            for slot, puff in enumerate(self.puffs):
                local = age - POP_ORDER.index(slot) * POP_STAGGER
                if puff is not None and 0 <= local < SMOKE_SECONDS:
                    draw_smoke(output, puff, local, foreground=False, burst=0.6, growth=0.3)
                    draw_smoke(output, puff, local, foreground=True, burst=0.5, growth=0.3)
        return output
