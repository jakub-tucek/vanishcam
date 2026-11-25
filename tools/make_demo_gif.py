"""Render docs/demo.gif: every effect played by the app's own renderers.

No camera is used: a glowing line figure does each gesture on a black stage. Run from the repo root:
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
from clone_effect import CLONE_SECONDS, CloneSquad  # noqa: E402
from snap_effect import SNAP_SECONDS, render_snap, snap_anchor  # noqa: E402

WIDTH, HEIGHT = 480, 270
FPS = 15
CAPTION = 30
OUTPUT = ROOT / "docs" / "demo.gif"


def backdrop() -> np.ndarray:
    """Black stage with a faint perspective grid (BGR)."""
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    horizon, cx = 150, WIDTH // 2
    for x in range(-WIDTH, 2 * WIDTH, 48):
        cv2.line(image, (cx + (x - cx) // 6, horizon), (x, HEIGHT), (70, 30, 60), 1, cv2.LINE_AA)
    for step in range(9):
        y = horizon + int((HEIGHT - horizon) * (step / 8) ** 1.8)
        cv2.line(image, (0, y), (WIDTH, y), (70, 30, 60), 1, cv2.LINE_AA)
    for x, y in ((40, 30), (110, 70), (390, 40), (450, 100), (300, 20), (180, 110), (70, 120)):
        cv2.circle(image, (x, y), 1, (150, 150, 150), -1)
    return image


# Joint offsets from the figure's centre line, in pixels at 270 px frame height.
# The figure faces the camera; "left"/"right" are screen sides.
IDLE = dict(l_elbow=(-30, 118), l_hand=(-38, 158), r_elbow=(30, 118), r_hand=(38, 158))
POSES = {
    "idle": IDLE,
    "v_sign": dict(IDLE, r_elbow=(44, 86), r_hand=(52, 46)),
    "palms": dict(l_elbow=(-34, 122), l_hand=(-5, 102), r_elbow=(34, 122), r_hand=(9, 102)),
    "forehead": dict(IDLE, r_elbow=(48, 90), r_hand=(31, 43)),
    "crossed": dict(l_elbow=(-32, 132), l_hand=(22, 86), r_elbow=(32, 132), r_hand=(-22, 86)),
    "snap": dict(IDLE, r_elbow=(48, 100), r_hand=(46, 62)),
    "seal": dict(l_elbow=(-36, 118), l_hand=(-14, 104), r_elbow=(36, 124), r_hand=(4, 116)),
}
HEAD, HIP = (0, 48), (0, 160)
LEGS = (((-16, 206), (-26, 252)), ((16, 206), (26, 252)))


def blend(pose_a: str, pose_b: str, amount: float) -> dict[str, tuple[float, float]]:
    amount = 0.5 - 0.5 * np.cos(np.pi * float(np.clip(amount, 0, 1)))
    a, b = POSES[pose_a], POSES[pose_b]
    return {k: (a[k][0] + (b[k][0] - a[k][0]) * amount, a[k][1] + (b[k][1] - a[k][1]) * amount)
            for k in a}


def figure(background: np.ndarray, joints: dict, color: tuple[int, int, int],
           detail: str = "") -> tuple[np.ndarray, np.ndarray]:
    """A glowing line figure over the backdrop, and its mask."""
    cx = WIDTH // 2
    lines = np.zeros((HEIGHT, WIDTH), np.uint8)

    def pt(offset):
        return int(cx + offset[0]), int(offset[1])

    cv2.circle(lines, pt(HEAD), 21, 255, 5, cv2.LINE_AA)
    cv2.line(lines, pt((0, 69)), pt(HIP), 255, 5, cv2.LINE_AA)
    for knee, foot in LEGS:
        cv2.polylines(lines, [np.array([pt(HIP), pt(knee), pt(foot)])], False, 255, 5, cv2.LINE_AA)
    cv2.line(lines, pt((-8, 84)), pt((8, 84)), 255, 5, cv2.LINE_AA)
    for side, shoulder in (("l", -8), ("r", 8)):
        arm = np.array([pt((shoulder, 84)), pt(joints[f"{side}_elbow"]),
                        pt(joints[f"{side}_hand"])])
        cv2.polylines(lines, [arm], False, 255, 5, cv2.LINE_AA)
    hand = np.array(pt(joints["r_hand"]))
    if detail == "v":
        for dx in (-9, 9):
            cv2.line(lines, tuple(hand), tuple(hand + (dx, -18)), 255, 4, cv2.LINE_AA)
    elif detail == "forehead":
        for dy in (-11, -4):
            cv2.line(lines, tuple(hand), tuple(hand + (-17, dy)), 255, 2, cv2.LINE_AA)
    elif detail == "palms":
        for side, dx in (("l", 7), ("r", -7)):
            wrist = np.array(pt(joints[f"{side}_hand"]))
            cv2.line(lines, tuple(wrist), tuple(wrist + (dx, -19)), 255, 3, cv2.LINE_AA)
    elif detail == "seal":
        left, right = np.array(pt(joints["l_hand"])), np.array(pt(joints["r_hand"]))
        cv2.line(lines, tuple(left), tuple(left + (26, 0)), 255, 3, cv2.LINE_AA)
        cv2.line(lines, tuple(right), tuple(right + (0, -24)), 255, 3, cv2.LINE_AA)
    elif detail == "snap":
        for offset in ((-6, -7), (2, -10)):
            cv2.line(lines, tuple(hand), tuple(hand + offset), 255, 3, cv2.LINE_AA)
    mask = cv2.GaussianBlur(cv2.dilate(lines, np.ones((7, 7), np.uint8)), (7, 7), 0)
    mask = mask.astype(np.float32) / 255
    glow = cv2.GaussianBlur(lines, (0, 0), 6).astype(np.float32) / 255
    core = lines.astype(np.float32) / 255
    tint = np.array(color, np.float32)
    frame = background.astype(np.float32)
    frame += glow[:, :, None] * tint * 1.4
    frame = frame * (1 - core[:, :, None]) + core[:, :, None] * (tint * 0.55 + 255 * 0.45)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame, mask


def snap_flash(scene: np.ndarray) -> list[np.ndarray]:
    center = np.array(POSES["snap"]["r_hand"]) + (WIDTH // 2 + 10, -12)
    frames = []
    for spark in (0.25, 0.65, 1):
        frame = scene.copy()
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            direction = np.array([np.cos(angle), np.sin(angle)])
            start = center + direction * (3 + 6 * spark)
            end = center + direction * (7 + 13 * spark)
            cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)),
                     (80, 255, 255), 2, cv2.LINE_AA)
        frames.append(frame)
    return frames


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


def gesture(background, pose: str, color, detail: str = "") -> list[np.ndarray]:
    """Idle, then the figure moves into the gesture and holds it."""
    clip = [figure(background, POSES["idle"], color)[0]] * round(0.3 * FPS)
    clip += [figure(background, blend("idle", pose, t / 0.45), color,
                    detail if t >= 0.45 else "")[0] for t in timeline(0.45)]
    return clip + [figure(background, POSES[pose], color, detail)[0]] * round(0.35 * FPS)


def relax(background, pose: str, color, detail: str = "") -> list[np.ndarray]:
    clip = [figure(background, POSES[pose], color, detail)[0]] * round(0.25 * FPS)
    return clip + [figure(background, blend(pose, "idle", t / 0.45), color)[0]
                   for t in timeline(0.45)] + [figure(background, POSES["idle"], color)[0]] * 4


def main() -> None:
    background = backdrop()
    clips = []

    cyan, magenta, lime, gold, red, pink = ((255, 230, 40), (255, 60, 230), (60, 255, 120),
                                            (40, 210, 255), (70, 70, 255), (190, 110, 255))

    frame, mask = figure(background, POSES["v_sign"], cyan, "v")
    field, noise = app.make_dissolve_fields(HEIGHT, WIDTH)
    clip = gesture(background, "v_sign", cyan, "v") + [
        app.apply_dissolve(frame, background, mask, min(1.0, t / 1.6), field, noise)
        for t in timeline(1.6)] + hold(background, 0.4)
    clips.append(captioned(clip, "1", "Normal", "V sign: dissolve"))

    frame, mask = figure(background, POSES["palms"], gold, "palms")
    sprite, cache = load_log(), {}
    anchor = substitution_anchor(mask)
    landed = background.copy()
    overlay_log(landed, sprite, landed_log_anchor(anchor, HEIGHT), cache=cache)
    clip = gesture(background, "palms", gold, "palms") + [
        render_substitution(frame, background, mask, sprite, anchor, max(0.04, t), cache)
        for t in timeline(NARUTO_SECONDS)] + hold(landed, 0.6)
    clips.append(captioned(clip, "2", "Naruto", "palms together: smoke, a log stays behind"))

    frame, mask = figure(background, POSES["v_sign"], magenta, "v")
    particles = ParticleEffect(frame, mask, seed=5)
    clip = gesture(background, "v_sign", magenta, "v") + [
        particles.render(background, t) for t in timeline(PARTICLE_SECONDS)] + hold(background, 0.3)
    clips.append(captioned(clip, "3", "Particles", "V sign: blown away in the wind"))

    frame, mask = figure(background, POSES["forehead"], lime, "forehead")
    teleport_out = GokuTeleport(frame, background, mask)
    teleport_in = GokuTeleport(frame, background, mask)
    slow = 3  # The teleport lasts 0.25 s; play it slower so it reads in a GIF.
    clip = (gesture(background, "forehead", lime, "forehead")
            + [teleport_out.render(background, t / slow, False) for t in timeline(TELEPORT_SECONDS * slow)]
            + hold(background, 0.5)
            + [teleport_in.render(background, t / slow, True) for t in timeline(TELEPORT_SECONDS * slow)]
            + relax(background, "forehead", lime, "forehead"))
    clips.append(captioned(clip, "4", "Goku", "two fingers on forehead: teleport out and back"))

    frame, mask = figure(background, POSES["crossed"], red)
    crows = ItachiCrowEffect(frame, mask)
    speed = 1.6  # Shorten the 4 s scatter for the GIF.
    clip = gesture(background, "crossed", red) + [
        crows.render(background, t * speed, False)
        for t in timeline(ITACHI_SECONDS / speed)] + hold(background, 0.3)
    clips.append(captioned(clip, "5", "Itachi", "arms crossed: scatter into crows"))

    frame, mask = figure(background, POSES["snap"], pink, "snap")
    cover = app.widen_mask(cv2.dilate(mask, np.ones((15, 15), np.uint8)))
    puff = snap_anchor(cover)
    clip = (gesture(background, "snap", pink, "snap") + snap_flash(frame)
            + [render_snap(frame, background, cover, puff, t, False) for t in timeline(SNAP_SECONDS)]
            + hold(background, 0.3) + snap_flash(background)
            + [render_snap(frame, background, cover, puff, t, True) for t in timeline(SNAP_SECONDS)]
            + relax(background, "snap", pink, "snap"))
    clips.append(captioned(clip, "6", "Snap", "finger snap: gone in smoke, snap again: back"))

    orange = (40, 140, 255)
    frame, mask = figure(background, POSES["seal"], orange, "seal")
    squad = CloneSquad(mask)
    clip = gesture(background, "seal", orange, "seal")
    clip += [squad.render(frame, mask, t, "appearing") for t in timeline(CLONE_SECONDS)]
    for t in timeline(1.6):  # The clones copy the live figure: wave, and they all wave.
        pose = blend("seal", "v_sign", 1 - abs(1 - t / 0.8))
        live, live_mask = figure(background, pose, orange)
        clip.append(squad.render(live, live_mask))
    clip += [squad.render(frame, mask) for _ in range(round(0.4 * FPS))]
    clip += [squad.render(frame, mask, t, "leaving") for t in timeline(CLONE_SECONDS)]
    clip += relax(background, "seal", orange, "seal")
    clips.append(captioned(clip, "7", "Clones", "crossed fingers: live shadow clones, again to dismiss"))

    frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for clip in clips for f in clip]
    samples = frames[::max(1, len(frames) // 24)]
    sheet = Image.new("RGB", (WIDTH, (HEIGHT + CAPTION) * len(samples)))
    for index, sample in enumerate(samples):
        sheet.paste(sample, (0, index * (HEIGHT + CAPTION)))
    palette = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    OUTPUT.parent.mkdir(exist_ok=True)
    # GIF delays are whole centiseconds; round cumulative times to avoid drift.
    ticks = [round(i * 100 / FPS) for i in range(len(frames) + 1)]
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=list(10 * np.diff(ticks)),
                   loop=0, optimize=True, disposal=1)
    print(f"{OUTPUT.relative_to(ROOT)}: {len(frames)} frames, "
          f"{len(frames) / FPS:.1f} s, {OUTPUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
