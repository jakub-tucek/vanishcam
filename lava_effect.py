"""Thumbs up and sink into lava: a molten pool rises, you go under, the pool drains away."""

from __future__ import annotations

from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np

LAVA_SECONDS = 4.4
RISE = (0.0, 0.6)          # Pool rises from the bottom edge to its surface level.
SINK = (0.45, 3.4)         # The person sinks below the surface.
DRAIN = (3.5, 4.4)         # Pool drains back down.
SURFACE_LEVEL = 0.8        # Pool surface, as a share of the frame height.
HEAT_BAND = 0.11           # Glow and shimmer above the surface, as a share of the frame height.

# Heat (0..1) to BGR: cooled crust, deep red, orange, yellow, white-hot.
_STOPS = ((0.0, (6, 4, 22)), (0.28, (8, 14, 105)), (0.48, (14, 52, 210)),
          (0.68, (30, 128, 255)), (0.86, (110, 214, 255)), (1.0, (215, 248, 255)))
PALETTE = np.stack([np.interp(np.linspace(0, 1, 256), [s for s, _ in _STOPS], [c[i] for _, c in _STOPS])
                    for i in range(3)], axis=1).astype(np.float32)


def _progress(age: float, window: tuple[float, float]) -> float:
    start, end = window
    amount = float(np.clip((age - start) / (end - start), 0.0, 1.0))
    return amount * amount * (3 - 2 * amount)


def _noise(height: int, width: int, rng: np.random.Generator, cell: int) -> np.ndarray:
    small = rng.random((height // cell + 3, width // cell + 3)).astype(np.float32)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def _fbm(height: int, width: int, rng: np.random.Generator, cell: int, octaves: int = 4) -> np.ndarray:
    """Fractal noise: several octaves of smooth noise, finer ones weaker."""
    total, weight, norm = np.zeros((height, width), np.float32), 1.0, 0.0
    for _ in range(octaves):
        total += _noise(height, width, rng, max(2, cell)) * weight
        norm += weight
        cell, weight = cell // 2, weight * 0.5
    return total / norm


@lru_cache(maxsize=2)
def lava_texture(height: int, width: int, seed: int = 77) -> tuple[np.ndarray, np.ndarray]:
    """(BGR colour, emission 0..1) of a tileable-in-x molten surface seen from above.

    Cooled crust plates (Voronoi cells) with rough lit surfaces, separated by cracks whose
    incandescent core fades through yellow and orange to red; crust edges facing a crack
    pick up its red glow.
    """
    rng = np.random.default_rng(seed)
    # Voronoi plates: label each pixel with its nearest seed point; plate borders are cracks.
    count = max(24, width * height // 2600)
    seeds = np.zeros((height, width), np.uint8)
    points = np.column_stack([rng.integers(0, height, count), rng.integers(0, width, count)])
    seeds[points[:, 0], points[:, 1]] = 1
    # Wrap horizontally so the texture scrolls seamlessly: label a tripled strip.
    wide = np.hstack([seeds, seeds, seeds])
    _, labels = cv2.distanceTransformWithLabels(1 - wide, cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_CCOMP)
    labels = labels[:, width:2 * width]
    border = np.zeros((height, width), np.uint8)
    border[:, :-1] |= (labels[:, :-1] != labels[:, 1:]).astype(np.uint8)
    border[:-1, :] |= (labels[:-1, :] != labels[1:, :]).astype(np.uint8)
    # Distort the straight Voronoi edges into ragged cracks.
    warp_x = (_fbm(height, width, rng, 48) - 0.5) * 26
    warp_y = (_fbm(height, width, rng, 48) - 0.5) * 26
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    border = cv2.remap(border * 255, grid_x + warp_x, grid_y + warp_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_WRAP)
    distance = cv2.distanceTransform((border < 128).astype(np.uint8), cv2.DIST_L2, 5)
    # Crack width varies along its length: wide molten rivers to hairline seams.
    width_map = 1.2 + _fbm(height, width, rng, 64) ** 2 * 9
    glow = np.exp(-distance / width_map)
    rough = _fbm(height, width, rng, 32, octaves=5)
    emission = np.clip(glow * (0.75 + 0.5 * rough) - 0.05, 0, 1) ** 1.3
    # Crust: dark basalt, shaded by its own roughness as if lit from the cracks below.
    height_map = cv2.GaussianBlur(rough, (0, 0), 1.2)
    gx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
    shade = np.clip(0.55 + (gx * 0.6 - gy * 0.9) * 9, 0.15, 1.1)
    crust = (np.array([22, 24, 30], np.float32) + rough[:, :, None] * np.array([28, 30, 38], np.float32))
    crust = crust * shade[:, :, None]
    # Red glow creeping onto crust edges next to cracks.
    rim = np.exp(-distance / (width_map * 3.5))[:, :, None]
    crust += rim * np.array([6, 30, 150], np.float32)
    hot = PALETTE[(np.clip(emission * 1.08, 0, 1) * 255).astype(np.uint8)]
    mix = np.clip(emission * 1.6, 0, 1)[:, :, None]
    colour = crust * (1 - mix) + hot * mix
    return colour.astype(np.float32), emission.astype(np.float32)


class LavaEffect:
    """Renders one sinking (or, played backwards, rising) of the person captured at the gesture."""

    def __init__(self, frame: np.ndarray, mask: np.ndarray, seed: int = 77):
        height, width = mask.shape[:2]
        rows = np.flatnonzero(mask.max(axis=1) > 0.5)
        if rows.size == 0:
            raise ValueError("No person to sink.")
        columns = np.flatnonzero(mask.max(axis=0) > 0.5)
        self.frame = frame.copy()
        self.mask = mask.astype(np.float32)
        self.size = (width, height)
        self.surface = int(height * SURFACE_LEVEL)
        self.sink_depth = self.surface - int(rows[0]) + 12
        self.body = (float(columns.mean()), max(8.0, (columns[-1] - columns[0]) / 2))
        self.band = max(12, int(height * HEAT_BAND))
        rng = np.random.default_rng(seed)
        self.colour, self.emission = lava_texture(height, width, seed)
        # Far texture is squeezed many times over; a blurred copy keeps its cracks from flickering.
        self.colour_far = cv2.GaussianBlur(self.colour, (0, 0), 2.5)
        self.emission_far = cv2.GaussianBlur(self.emission, (0, 0), 2.5)
        self.warp = (_noise(height, width, rng, max(12, width // 12)) - 0.5) * (width / 40)
        count = 70
        self.embers = np.column_stack([rng.random(count) * width, rng.random(count),
                                       rng.random(count) * 0.6 + 0.35, rng.random(count)])
        self.bubbles = np.column_stack([rng.random(9) * width, rng.random(9),
                                        rng.random(9) * 0.5 + 0.5, rng.random(9) * 0.5 + 0.7])
        self.x = np.arange(width, dtype=np.float32)

    def _surface_line(self, level: float, age: float, dipping: float) -> np.ndarray:
        x, (cx, half) = self.x, self.body
        wave = 4 * np.sin(x / 47 + age * 2.3) + 2.5 * np.sin(x / 19 - age * 3.7)
        # The pool heaps up a little around the body while it goes in.
        heap = 7 * dipping * np.exp(-((x - cx) / (half + 14)) ** 2)
        return (level + wave - heap).astype(np.float32)

    def _lava(self, age: float, top: int, level: float, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width, height = self.size
        rows = np.arange(top, height, dtype=np.float32)[:, None]
        # Depth below each column's own surface, so the heap around the body is textured too.
        depth = np.maximum(rows - line[None, :], 0)
        # Ground-plane perspective: the horizon sits a little above the surface, so texture
        # is strongly squeezed at the far edge and close to life size at the bottom of frame.
        horizon = height * 0.08
        near_depth = height * 0.28
        scale = near_depth / (depth + horizon)            # Texture pixels per screen pixel.
        tex_y = near_depth ** 2 / (depth + horizon)
        tex_y = (near_depth ** 2 / horizon - tex_y) % (height - 1)
        drift = float(age) * width * 0.02
        centre = width / 2
        map_x = ((self.x[None, :] - centre) * scale + centre + drift).astype(np.float32)
        map_y = np.broadcast_to(tex_y, map_x.shape).astype(np.float32)
        warp = cv2.remap(self.warp, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        map_x = (map_x + warp * np.sin(age * 0.7)).astype(np.float32)

        def sample(texture):
            return cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        far = np.clip((scale - 1.5) / 3, 0, 1)
        lava = sample(self.colour) * (1 - far[:, :, None]) + sample(self.colour_far) * far[:, :, None]
        heat = sample(self.emission) * (1 - far) + sample(self.emission_far) * far
        # The molten cracks breathe; the pool is hotter right at the surface where it churns.
        pulse = 1 + 0.12 * np.sin(age * 4 + map_x / 45 + map_y / 30)
        near = np.exp(-(rows - line[None, :]).clip(0) / (self.band * 0.5))
        heat = np.clip(heat * pulse + near * 0.15, 0, 1)
        lava = lava * pulse[:, :, None] + PALETTE[(heat * 255).astype(np.uint8)] * (near * 0.3)[:, :, None]
        return lava, heat

    def render(self, background: np.ndarray, age: float, appearing: bool = False,
               frame: np.ndarray | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        """Frame at age seconds; appearing plays the same sequence backwards.

        frame and mask are the live person, so they keep moving while they sink or rise;
        without them the snapshot from the gesture is used.
        """
        frame = self.frame if frame is None else frame
        mask = self.mask if mask is None else mask
        if appearing:
            age = LAVA_SECONDS - age
        width, height = self.size
        rise = _progress(age, RISE) - _progress(age, DRAIN)
        sunk = _progress(age, SINK)
        drop = int(round(self.sink_depth * sunk))
        dipping = float(rise > 0 and 0 < drop < self.sink_depth)
        level = height + 8 - (height + 8 - self.surface) * rise
        line = self._surface_line(level, age, dipping)
        rows = np.arange(height, dtype=np.float32)[:, None]

        output = background.astype(np.float32)
        if drop < self.sink_depth:
            matrix = np.float32([[1, 0, 0], [0, 1, drop]])
            person = cv2.warpAffine(frame, matrix, (width, height)).astype(np.float32)
            alpha = cv2.warpAffine(mask.astype(np.float32), matrix, (width, height)) * (rows < line[None, :])
            if rise > 0:
                # Close to the surface the body glows red-hot, then chars just above it.
                heat = np.clip(1 - (line[None, :] - rows) / self.band, 0, 1)[:, :, None]
                hot = np.array([30, 110, 255], np.float32)
                person = person * (1 - 0.6 * heat) + hot * 0.6 * heat * (1 - 0.5 * heat ** 4)
            output = output * (1 - alpha[:, :, None]) + person * alpha[:, :, None]

        if rise <= 0:
            return np.clip(output, 0, 255).astype(np.uint8)

        # The pool lights the room: an orange wash from below, fading with height, that
        # tints what is there (multiplied) rather than painting over it.
        flicker = 1 + 0.06 * np.sin(age * 17) + 0.04 * np.sin(age * 31 + 1)
        reach = height * 0.16
        start = max(0, int(line.min() - reach * 4.5))  # Below 1% of full strength above this.
        lit = (np.exp(-np.clip(line[None, :] - rows[start:], 0, None) / reach) * rise * flicker)[:, :, None]
        tint = np.array([0.05, 0.3, 0.8], np.float32)
        output[start:] = output[start:] * (1 + lit * tint) + lit * np.array([0, 6, 22], np.float32)

        top = max(0, int(line.min()) - self.band)
        region = output[top:]
        local = rows[top:] - line[None, :]
        above = local < 0

        # Heat shimmer: the air just above the pool wobbles.
        shimmer = np.clip(1 + local / self.band, 0, 1) * above
        offset = 2.2 * shimmer * np.sin(rows[top:] / 5 + age * 11 + self.x[None, :] / 37)
        map_x = (self.x[None, :] + offset).astype(np.float32)
        map_y = (np.broadcast_to(rows[top:], map_x.shape) - top).astype(np.float32)
        region[:] = cv2.remap(region, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        glow = np.clip(1 + local / self.band, 0, 1) ** 2 * above
        region += glow[:, :, None] * np.array([10, 45, 140], np.float32) * 0.5 * rise

        lava, heat = self._lava(age, top, level, line)
        inside = (~above)[:, :, None]
        region[:] = np.where(inside, lava, region)
        rim = np.clip(1 - np.abs(local) / 2.5, 0, 1)[:, :, None]
        region[:] = region * (1 - rim) + np.array([150, 240, 255], np.float32) * rim

        # Bloom: the hottest lava bleeds light into its surroundings.
        small = cv2.resize(np.where(inside, lava * (heat[:, :, None] > 0.7), 0).astype(np.float32),
                           (width // 4, max(1, region.shape[0] // 4)), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (width, region.shape[0]))
        region += bloom * 0.55

        canvas = np.clip(output, 0, 255).astype(np.uint8)
        self._details(canvas, age, line, dipping, rise)
        return canvas

    def _details(self, canvas: np.ndarray, age: float, line: np.ndarray, dipping: float,
                 rise: float) -> None:
        width, height = self.size
        cx, half = self.body
        # Ripples spread from where the body breaks the surface: added light, brighter in
        # front of the body than behind it.
        if dipping:
            y = int(line[int(np.clip(cx, 0, width - 1))]) + 3
            light = np.zeros((height, width, 3), np.uint8)
            for ring in range(3):
                phase = (age * 0.9 + ring / 3) % 1.0
                radius = int(half + 6 + phase * width * 0.16)
                strength = (1 - phase) ** 1.5
                axes = (radius, max(2, radius // 10))
                cv2.ellipse(light, (int(cx), y), axes, 0, 0, 180,
                            tuple(int(c * strength) for c in (40, 140, 255)), 2, cv2.LINE_AA)
                cv2.ellipse(light, (int(cx), y), axes, 0, 180, 360,
                            tuple(int(c * strength * 0.4) for c in (40, 140, 255)), 1, cv2.LINE_AA)
            cv2.add(canvas, cv2.GaussianBlur(light, (3, 3), 0), dst=canvas)
        # Bubbles swell on the surface and pop.
        for x, offset, speed, size in self.bubbles:
            phase = (age * speed + offset) % 1.0
            px = int(x)
            if rise < 0.6 or phase > 0.85:
                continue
            radius = int(size * (2 + phase * height * 0.02))
            y = int(line[px]) + 1
            cv2.ellipse(canvas, (px, y), (radius, int(radius * 0.8)), 0, 180, 360,
                        (40, 150, 255), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (px, y), (radius, int(radius * 0.8)), 0, 200, 340,
                        (170, 245, 255), 1, cv2.LINE_AA)
        # Embers drift up and fade.
        for x, offset, speed, flicker in self.embers:
            lift = (age * speed + offset) % 1.0
            px = int(x + 8 * np.sin(age * 2.5 + offset * 11))
            if not 0 <= px < width:
                continue
            y = int(line[px] - 4 - lift * self.band * 3)
            if not 0 <= y < height:
                continue
            fade = (1 - lift) * rise * (0.6 + 0.4 * np.sin(age * 13 + flicker * 20))
            color = tuple(int(c * fade) for c in (110, 215, 255))
            base = canvas[y, px].astype(np.int32)
            cv2.circle(canvas, (px, y), 1 + (flicker > 0.75), tuple(
                int(min(255, b + c)) for b, c in zip(base, color)), -1, cv2.LINE_AA)


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
