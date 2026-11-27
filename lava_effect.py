"""Thumbs up and sink into lava: a molten pool rises, you go under, the pool drains away."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

LAVA_SECONDS = 4.4
RISE = (0.0, 0.6)          # Pool rises from the bottom edge to its surface level.
SINK = (0.45, 3.4)         # The person sinks below the surface.
DRAIN = (3.5, 4.4)         # Pool drains back down.
SURFACE_LEVEL = 0.8        # Pool surface, as a share of the frame height.
HEAT_BAND = 60             # Pixels above the surface lit by the glow.


def _progress(age: float, window: tuple[float, float]) -> float:
    start, end = window
    amount = float(np.clip((age - start) / (end - start), 0.0, 1.0))
    return amount * amount * (3 - 2 * amount)


def _noise(height: int, width: int, seed: int, cell: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.random((height // cell + 2, width // cell + 2)).astype(np.float32)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


class LavaEffect:
    """Renders one sinking (or, played backwards, rising) of the person captured at the gesture."""

    def __init__(self, frame: np.ndarray, mask: np.ndarray, seed: int = 77):
        height, width = mask.shape[:2]
        rows = np.flatnonzero(mask.max(axis=1) > 0.5)
        if rows.size == 0:
            raise ValueError("No person to sink.")
        self.frame = frame.copy()
        self.mask = mask.astype(np.float32)
        self.size = (width, height)
        self.surface = int(height * SURFACE_LEVEL)
        self.sink_depth = self.surface - int(rows[0]) + 12
        # Two drifting noise layers make the molten surface churn; a coarse one marks crust.
        self.flow = [_noise(height, width * 2, seed, 24), _noise(height, width * 2, seed + 1, 9)]
        self.crust = _noise(height, width * 2, seed + 2, 40)
        rng = np.random.default_rng(seed + 3)
        self.embers = np.column_stack([rng.random(60) * width, rng.random(60), rng.random(60) * 0.8 + 0.4])
        x = np.arange(width, dtype=np.float32)
        self.x = x

    def _surface_line(self, level: float, age: float) -> np.ndarray:
        x = self.x
        wave = 5 * np.sin(x / 41 + age * 2.7) + 3 * np.sin(x / 17 - age * 4.1)
        return (level + wave).astype(np.float32)

    def _lava(self, age: float, top: int) -> np.ndarray:
        width, height = self.size
        shift = int(age * 38) % width
        slow = self.flow[0][top:, shift:shift + width]
        fast = self.flow[1][top:, (width - shift * 2 % width):(width - shift * 2 % width) + width]
        heat = np.clip(slow * 0.65 + fast * 0.45 - 0.05, 0, 1)
        crust = self.crust[top:, (shift // 2):(shift // 2) + width] > 0.68
        # Palette in BGR: deep red -> orange -> yellow-white.
        red = np.array([10, 20, 150], np.float32)
        orange = np.array([20, 110, 255], np.float32)
        yellow = np.array([120, 235, 255], np.float32)
        low = heat[:, :, None] < 0.5
        mix_low = red + (orange - red) * (heat[:, :, None] * 2)
        mix_high = orange + (yellow - orange) * ((heat[:, :, None] - 0.5) * 2)
        lava = np.where(low, mix_low, mix_high)
        lava[crust] *= 0.35
        return lava

    def render(self, background: np.ndarray, age: float, appearing: bool = False) -> np.ndarray:
        """Frame at age seconds; appearing plays the same sequence backwards."""
        if appearing:
            age = LAVA_SECONDS - age
        width, height = self.size
        rise = _progress(age, RISE) - _progress(age, DRAIN)
        level = height + 8 - (height + 8 - self.surface) * rise
        line = self._surface_line(level, age)
        rows = np.arange(height, dtype=np.float32)[:, None]

        output = background.astype(np.float32)
        drop = int(round(self.sink_depth * _progress(age, SINK)))
        if drop < self.sink_depth:
            matrix = np.float32([[1, 0, 0], [0, 1, drop]])
            person = cv2.warpAffine(self.frame, matrix, (width, height)).astype(np.float32)
            alpha = cv2.warpAffine(self.mask, matrix, (width, height))
            alpha = alpha * (rows < line[None, :])
            # The body glows hotter the closer it is to the surface.
            heat = np.clip(1 - (line[None, :] - rows) / HEAT_BAND, 0, 1) * (rise > 0)
            person = person * (1 - 0.5 * heat[:, :, None]) + np.array([30, 120, 255]) * 0.5 * heat[:, :, None]
            output = output * (1 - alpha[:, :, None]) + person * alpha[:, :, None]

        if rise > 0:
            top = max(0, int(line.min()) - HEAT_BAND)
            below = rows[top:] >= line[None, :]
            glow = np.clip(1 - (line[None, :] - rows[top:]) / HEAT_BAND, 0, 1) * ~below
            region = output[top:]
            region += glow[:, :, None] ** 2 * np.array([20, 90, 230], np.float32) * 0.8 * rise
            lava = self._lava(age, top)
            rim = np.abs(rows[top:] - line[None, :]) < 2.5
            lava[rim] = np.array([170, 245, 255], np.float32)
            region[below] = lava[below]
            for x, phase, speed in self.embers:
                lift = ((age * speed + phase) % 1.0)
                y = int(level - lift * HEAT_BAND * 2.2)
                px = int(x + 6 * np.sin(age * 3 + phase * 9))
                if 0 <= px < width and top <= y < height and y < line[min(px, width - 1)]:
                    cv2.circle(output, (px, y), 1, (90, 200, 255), -1, cv2.LINE_AA)
        return np.clip(output, 0, 255).astype(np.uint8)


# Calibrated on labelled recordings: every held thumbs up (raised or at chest height) scores
# 0.6-0.8 with MediaPipe's gesture model; fists, sideways or downward thumbs, pointing, OK,
# V and open hands never pass. Together with a 0.3 s hold.
THUMB_UP_MIN = 0.55


class ThumbsUpRecognizer:
    """MediaPipe's canned gesture model, limited to Thumb_Up."""

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.last_score = 0.0
        self.last_hand_count = 0

    def recognize_thumb(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.recognizer.recognize_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        self.last_score = max((category.score for categories in result.gestures
                               for category in categories if category.category_name == "Thumb_Up"),
                              default=0.0)
        return self.last_score > THUMB_UP_MIN

    def describe(self) -> str:
        if not self.last_hand_count:
            return "No hand in view"
        return f"Thumbs up: {self.last_score:.0%} (needs over {THUMB_UP_MIN:.0%})"

    def close(self) -> None:
        self.recognizer.close()
