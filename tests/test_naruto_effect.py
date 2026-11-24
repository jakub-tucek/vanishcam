import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import vanish_cam as app
from naruto_effect import (
    EFFECT_SECONDS, analyze_hand_seal, falling_log_anchor, joined_hands_from_landmarks,
    landed_log_anchor, load_log, overlay_log, render_substitution, substitution_anchor,
)
from test_video_output import arguments


class JoinedHandsTests(unittest.TestCase):
    @staticmethod
    def gate():
        return app.GokuGestureGate(hold_seconds=app.NARUTO_HOLD_SECONDS)

    def test_palms_must_stay_together_before_vanishing(self):
        state, gate = app.AppState(phase="ready"), self.gate()
        for now in (1.0, 1.1, 1.2):
            self.assertFalse(app.start_naruto_on_join(state, gate, True, now))
        self.assertTrue(app.start_naruto_on_join(state, gate, True, 1.3))
        self.assertEqual(state.phase, "naruto_vanishing")
        self.assertEqual(state.vanish_started, 1.3)

    def test_a_brief_touch_never_triggers(self):
        state, gate = app.AppState(phase="ready"), self.gate()
        for now, joined in ((1.0, True), (1.1, False), (1.4, False), (1.5, True), (1.6, True)):
            self.assertFalse(app.start_naruto_on_join(state, gate, joined, now))
        self.assertEqual(state.phase, "ready")

    def test_hands_cut_off_by_the_frame_edge_are_ignored(self):
        lowered = [[SimpleNamespace(x=point.x, y=point.y + 0.4) for point in self.hand(center)]
                   for center in (0.45, 0.55)]  # Wrists below the bottom edge, as on a knee.
        self.assertFalse(joined_hands_from_landmarks(lowered, 640, 480))

    @staticmethod
    def hand(center_x, *, spread=0.004, curl=False, top=0.47):
        """One hand of a seal, seen edge-on: wrist low, straight closed fingers pointing up."""
        points = [SimpleNamespace(x=center_x, y=0.75)]
        offsets = (-1.5 * spread, -0.5 * spread, 0.5 * spread, 1.5 * spread)
        points += [SimpleNamespace(x=center_x, y=0.70 - 0.02 * i) for i in range(4)]  # Thumb.
        for offset in offsets:
            tip_y = 0.62 if curl else top
            for y in (0.60, 0.55, 0.51, tip_y):
                points.append(SimpleNamespace(x=center_x + offset * (0.60 - y) / (0.60 - top), y=y))
        return points

    def seal(self, **options):
        return [self.hand(0.497, **options), self.hand(0.503, **options)]

    def test_palms_together_fingers_straight_closed_and_up_is_the_seal(self):
        seal = analyze_hand_seal(self.seal(), 640, 480)
        self.assertTrue(seal.ok, seal)
        self.assertTrue(joined_hands_from_landmarks(self.seal(), 640, 480))

    def test_each_part_of_the_seal_is_required(self):
        cases = {
            "fingers_together": self.seal(spread=0.06),
            "fingers_straight": self.seal(curl=True),
            "palms_together": [self.hand(0.35), self.hand(0.65)],
        }
        for failing, hands in cases.items():
            with self.subTest(failing):
                seal = analyze_hand_seal(hands, 640, 480)
                self.assertFalse(seal.ok)
                self.assertFalse(getattr(seal, failing))
        sideways = [[SimpleNamespace(x=0.5 + (p.y - 0.6), y=0.6 + (p.x - 0.5)) for p in hand]
                    for hand in self.seal()]
        self.assertFalse(analyze_hand_seal(sideways, 640, 480).pointing_up)
        self.assertEqual(analyze_hand_seal([self.hand(0.5)], 640, 480).hands, 1)

    def test_3d_hand_shape_overrides_an_edge_on_image(self):
        splayed = [[SimpleNamespace(x=p.x, y=p.y, z=0.0) for p in self.hand(0.5, spread=0.06)]
                   for _ in range(2)]
        seal = analyze_hand_seal(self.seal(), 640, 480, world_hands=splayed)
        self.assertFalse(seal.fingers_together)

    def test_feedback_names_what_is_missing(self):
        text = analyze_hand_seal(self.seal(spread=0.06), 640, 480).describe()
        self.assertIn("together: no", text)
        self.assertIn("palms touching: yes", text)
        self.assertIn("need both", analyze_hand_seal([self.hand(0.5)], 640, 480).describe())

    def test_rotated_views_map_back_to_the_real_frame(self):
        height, width = 48, 64
        row, column = 30, 50  # Inside both crops.
        for rotation in range(4):
            for crop in ((0.0, 0.0, 1.0, 1.0), (0.125, 0.25, 0.875, 1.0)):
                with self.subTest(rotation=rotation, crop=crop):
                    left, top, right, bottom = crop
                    frame = np.zeros((height, width), np.uint8)
                    frame[row, column] = 255
                    region = frame[int(height * top):int(height * bottom),
                                   int(width * left):int(width * right)]
                    rotated = np.rot90(region, rotation)
                    r, c = np.argwhere(rotated == 255)[0]
                    point = SimpleNamespace(x=(c + 0.5) / rotated.shape[1],
                                            y=(r + 0.5) / rotated.shape[0])
                    mapped = app.view_to_frame([[point]], rotation, crop)[0][0]
                    self.assertAlmostEqual(mapped.x, (column + 0.5) / width, places=6)
                    self.assertAlmostEqual(mapped.y, (row + 0.5) / height, places=6)

    def test_recognizer_uses_two_hands_in_the_current_frame(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=self.seal(), hand_world_landmarks=None)
        recognizer = app.MultiViewHandLandmarker(primary, alternate)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(recognizer.recognize_joined_hands(image, 100))
        self.assertEqual(recognizer.last_hand_count, 2)
        self.assertTrue(recognizer.last_seal.ok)
        alternate.detect.assert_not_called()

    def test_hand_model_recovers_the_seal_in_a_rotated_view(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[self.hand(0.497)], hand_world_landmarks=None)
        # The second retry view is the full frame turned clockwise (np.rot90 k=3).
        turned = [[SimpleNamespace(x=1 - p.y, y=p.x) for p in hand] for hand in self.seal()]
        alternate.detect.side_effect = [
            SimpleNamespace(hand_landmarks=[], hand_world_landmarks=None),
            SimpleNamespace(hand_landmarks=turned, hand_world_landmarks=None),
        ]
        recognizer = app.MultiViewHandLandmarker(primary, alternate)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(recognizer.recognize_joined_hands(image, 100))
        self.assertEqual(recognizer.preferred_view, ("full", 3))
        self.assertEqual(recognizer.last_hand_count, 2)


class EffectTests(unittest.TestCase):
    def test_generated_asset_has_real_transparency(self):
        sprite = load_log()
        self.assertEqual(sprite.shape[2], 4)
        self.assertEqual(int(sprite[:, :, 3].min()), 0)
        self.assertEqual(int(sprite[:, :, 3].max()), 255)

    def test_smoke_animation_leaves_log_behind(self):
        frame = np.full((160, 240, 3), 130, np.uint8)
        background = np.full_like(frame, 35)
        mask = np.ones(frame.shape[:2], np.float32)
        sprite = load_log()
        anchor = (120, 85, 100)
        early = render_substitution(frame, background, mask, sprite, anchor, 0.04)
        middle = render_substitution(frame, background, mask, sprite, anchor, 0.55)
        final = render_substitution(frame, background, mask, sprite, anchor, EFFECT_SECONDS)
        self.assertTrue(np.any(early != frame))
        self.assertTrue(np.any(middle != frame))
        self.assertTrue(np.any(middle != background))
        self.assertTrue(np.any(final != background))
        np.testing.assert_array_equal(final[0, 0], background[0, 0])

    def test_log_anchor_tracks_person_region(self):
        mask = np.zeros((160, 240), np.float32)
        mask[20:130, 140:210] = 1.0
        x, y, size = substitution_anchor(mask)
        self.assertGreater(x, 140)
        self.assertGreater(size, 35)
        self.assertTrue(20 < y < 130)

    def test_log_falls_then_stays_on_floor(self):
        anchor = (120, 80, 90)
        ys = [falling_log_anchor(anchor, 240, age)[1] for age in (0.1, 0.3, 0.5, 0.7)]
        self.assertEqual(ys[0], anchor[1])
        self.assertGreater(ys[2], ys[1])
        self.assertGreater(ys[3], ys[2])
        floor = landed_log_anchor(anchor, 240)
        self.assertGreater(floor[1], ys[3])
        background = np.full((240, 240, 3), 35, np.uint8)
        output = render_substitution(background, background, np.zeros((240, 240), np.float32),
                                     load_log(), anchor, EFFECT_SECONDS)
        expected = background.copy()
        overlay_log(expected, load_log(), floor)
        np.testing.assert_array_equal(output, expected)


class NarutoVideoLoopTests(unittest.TestCase):
    def test_effect_is_sent_to_preview_and_virtual_camera(self):
        clock = [0.0]
        times = iter((0.1, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.0, 6.2, 6.3))
        frames = []

        def read():
            clock[0] = next(times)
            frames.append(np.full((80, 120, 3), int(clock[0] * 30) % 250, np.uint8))
            return True, frames[-1]

        capture = Mock()
        capture.read.side_effect = read
        segmenter, recognizer = Mock(), Mock()
        recognizer.recognize_joined_hands.return_value = True
        sound = Mock()
        virtual = Mock(width=120, height=80)
        preview = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(mode="naruto")))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            mocks.enter_context(patch.object(app, "VanishSound", return_value=sound))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=virtual))
            mocks.enter_context(patch.object(app, "person_mask", return_value=np.ones((80, 120), np.float32)))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=lambda: clock[0]))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: preview.append(frame.copy())))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=[0] * 9 + [ord("q")]))
            mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("Overlay vietato")))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
        for sent, shown in zip(virtual.send.call_args_list, preview):
            np.testing.assert_array_equal(sent.args[0], shown)
        # Palms held from 4.2: still live at 4.4, the smoke starts once 0.3 s have passed.
        np.testing.assert_array_equal(preview[4], frames[4])
        self.assertTrue(np.any(preview[6] != frames[6]))
        self.assertTrue(np.any(preview[9] != frames[9]))  # The log stays after the smoke.
        sound.play.assert_called_once()
        sound.close.assert_called_once()

if __name__ == "__main__":
    unittest.main()
