"""Render docs/demo.gif: every effect played by the app's own renderers on a drawn scene.

No camera is used; the person is a simple drawn figure. Run from the repo root:
    .venv/bin/python tools/make_demo_gif.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vanish_cam as app  # noqa: E402
from goku_effect import GokuTeleport, TELEPORT_SECONDS  # noqa: E402
from itachi_effect import EFFECT_SECONDS as ITACHI_SECONDS, ItachiCrowEffect  # noqa: E402
from naruto_effect import (  # noqa: E402
    EFFECT_SECONDS as NARUTO_SECONDS, landed_log_anchor, load_log, overlay_log,
    render_substitution, substitution_anchor,
)
from particles_effect import PARTICLE_SECONDS, ParticleEffect  # noqa: E402
from snap_effect import SNAP_SECONDS, render_snap, snap_anchor  # noqa: E402

WIDTH, HEIGHT = 480, 270
FPS = 15
CAPTION = 30
OUTPUT = ROOT / "docs" / "demo.gif"


def room() -> np.ndarray:
    """A plain home-office wall: window, shelf, plant, desk edge (BGR)."""
    image = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    shade = np.linspace(0, 1, HEIGHT)[:, None]
    image[:] = (np.array([178, 196, 214]) * (1 - 0.15 * shade))[:, :, None].transpose(0, 2, 1)
    cv2.rectangle(image, (24, 34), (140, 150), (120, 92, 70), -1)
    cv2.rectangle(image, (30, 40), (134, 144), (235, 214, 170), -1)
    cv2.line(image, (82, 40), (82, 144), (120, 92, 70), 4)
    cv2.line(image, (30, 92), (134, 92), (120, 92, 70), 4)
    cv2.rectangle(image, (345, 70), (455, 78), (70, 98, 130), -1)
    for x, height, color in ((352, 30, (60, 60, 170)), (366, 26, (150, 110, 50)),
                             (378, 32, (70, 140, 90)), (392, 22, (40, 150, 200))):
        cv2.rectangle(image, (x, 70 - height), (x + 11, 70), color, -1)
    cv2.rectangle(image, (420, 52), (444, 70), (90, 120, 170), -1)
    cv2.ellipse(image, (432, 52), (18, 12), 0, 180, 360, (70, 150, 80), -1)
    cv2.rectangle(image, (0, 228), (WIDTH, HEIGHT), (80, 108, 140), -1)
    cv2.rectangle(image, (0, 228), (WIDTH, 233), (60, 84, 112), -1)
    return cv2.GaussianBlur(image, (3, 3), 0)


def person(background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A drawn person at a desk: the frame with them in it, and their mask."""
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    frame = background.copy()
    cx = WIDTH // 2
    body = np.array([[cx - 88, 228], [cx - 74, 150], [cx - 40, 128], [cx + 40, 128],
                     [cx + 74, 150], [cx + 88, 228]], np.int32)
    shirt = (150, 96, 52)
    cv2.fillPoly(frame, [body], shirt)
    cv2.fillPoly(mask, [body], 1)
    cv2.rectangle(frame, (cx - 13, 108), (cx + 13, 132), (120, 160, 215), -1)
    cv2.rectangle(mask, (cx - 13, 108), (cx + 13, 132), 1, -1)
    cv2.ellipse(frame, (cx, 128), (24, 10), 0, 0, 180, (120, 160, 215), -1)
    cv2.ellipse(frame, (cx, 84), (34, 40), 0, 0, 360, (140, 180, 230), -1)
    cv2.ellipse(mask, (cx, 84), (34, 40), 0, 0, 360, 1, -1)
    cv2.ellipse(frame, (cx, 70), (37, 30), 0, 180, 360, (40, 52, 70), -1)
    cv2.ellipse(mask, (cx, 70), (37, 30), 0, 180, 360, 1, -1)
    for dx in (-13, 13):
        cv2.circle(frame, (cx + dx, 88), 3, (50, 40, 40), -1)
    cv2.ellipse(frame, (cx, 102), (10, 5), 0, 10, 170, (70, 80, 150), 2)
    cv2.line(frame, (cx, 140), (cx, 228), (120, 76, 40), 2)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    frame = (frame * soft[:, :, None] + background * (1 - soft[:, :, None])).astype(np.uint8)
    return frame, soft


def timeline(seconds: float) -> np.ndarray:
    return np.arange(0, seconds + 1e-6, 1 / FPS)


def hold(image: np.ndarray, seconds: float) -> list[np.ndarray]:
    return [image] * max(1, round(seconds * FPS))


def captioned(frames: list[np.ndarray], key: str, name: str, gesture: str) -> list[np.ndarray]:
    out = []
    for frame in frames:
        canvas = np.full((HEIGHT + CAPTION, WIDTH, 3), 28, np.uint8)
        canvas[:HEIGHT] = frame
        cv2.rectangle(canvas, (10, HEIGHT + 6), (30, HEIGHT + 24), (90, 200, 255), -1)
        cv2.putText(canvas, key, (15, HEIGHT + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (28, 28, 28), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, name, (40, HEIGHT + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, gesture, (140, HEIGHT + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (190, 190, 190), 1, cv2.LINE_AA)
        out.append(canvas)
    return out


def main() -> None:
    background = room()
    frame, mask = person(background)
    clips = []

    field, noise = app.make_dissolve_fields(HEIGHT, WIDTH)
    clip = hold(frame, 0.5) + [
        app.apply_dissolve(frame, background, mask, min(1.0, t / 2.0), field, noise)
        for t in timeline(2.0)] + hold(background, 0.5)
    clips.append(captioned(clip, "1", "Normal", "V sign: dissolve"))

    sprite, cache = load_log(), {}
    anchor = substitution_anchor(mask)
    landed = background.copy()
    overlay_log(landed, sprite, landed_log_anchor(anchor, HEIGHT), cache=cache)
    clip = hold(frame, 0.5) + [
        render_substitution(frame, background, mask, sprite, anchor, max(0.04, t), cache)
        for t in timeline(NARUTO_SECONDS)] + hold(landed, 0.6)
    clips.append(captioned(clip, "2", "Naruto", "palms together: smoke, a log stays behind"))

    particles = ParticleEffect(frame, mask, seed=5)
    clip = hold(frame, 0.5) + [particles.render(background, t)
                               for t in timeline(PARTICLE_SECONDS)] + hold(background, 0.4)
    clips.append(captioned(clip, "3", "Particles", "V sign: blown away in the wind"))

    teleport_out = GokuTeleport(frame, background, mask)
    teleport_in = GokuTeleport(frame, background, mask)
    slow = 3  # The teleport lasts 0.25 s; play it slower so it reads in a GIF.
    clip = (hold(frame, 0.5)
            + [teleport_out.render(background, t / slow, False) for t in timeline(TELEPORT_SECONDS * slow)]
            + hold(background, 0.6)
            + [teleport_in.render(background, t / slow, True) for t in timeline(TELEPORT_SECONDS * slow)]
            + hold(frame, 0.4))
    clips.append(captioned(clip, "4", "Goku", "two fingers on forehead: teleport out and back"))

    crows = ItachiCrowEffect(frame, mask)
    speed = 1.6  # Shorten the 4 s scatter for the GIF.
    clip = hold(frame, 0.4) + [crows.render(background, t * speed, False)
                               for t in timeline(ITACHI_SECONDS / speed)] + hold(background, 0.4)
    clips.append(captioned(clip, "5", "Itachi", "arms crossed: scatter into crows"))

    cover = app.widen_mask(mask)
    puff = snap_anchor(cover)
    clip = (hold(frame, 0.4)
            + [render_snap(frame, background, cover, puff, t, False) for t in timeline(SNAP_SECONDS)]
            + hold(background, 0.5)
            + [render_snap(frame, background, cover, puff, t, True) for t in timeline(SNAP_SECONDS)]
            + hold(frame, 0.5))
    clips.append(captioned(clip, "6", "Snap", "finger snap: gone in smoke, snap again: back"))

    frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for clip in clips for f in clip]
    samples = frames[::max(1, len(frames) // 24)]
    sheet = Image.new("RGB", (WIDTH, (HEIGHT + CAPTION) * len(samples)))
    for index, sample in enumerate(samples):
        sheet.paste(sample, (0, index * (HEIGHT + CAPTION)))
    palette = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    OUTPUT.parent.mkdir(exist_ok=True)
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=round(1000 / FPS),
                   loop=0, optimize=True, disposal=1)
    print(f"{OUTPUT.relative_to(ROOT)}: {len(frames)} frames, "
          f"{len(frames) / FPS:.1f} s, {OUTPUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
