import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import vanish_cam as app
from naruto_effect import (
    EFFECT_SECONDS, falling_log_anchor, joined_hands_from_landmarks,
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
    def hand(center_x, upward=True):
        points = [SimpleNamespace(x=center_x, y=0.53) for _ in range(21)]
        points[0] = SimpleNamespace(x=center_x, y=0.70)
        for index, offset in ((5, -0.045), (9, -0.015), (13, 0.015), (17, 0.045)):
            points[index] = SimpleNamespace(x=center_x + offset, y=0.54 if upward else 0.84)
        for index, offset in ((8, -0.045), (12, -0.015), (16, 0.015), (20, 0.045)):
            points[index] = SimpleNamespace(x=center_x + offset, y=0.34 if upward else 1.04)
        return points

    def test_two_nearby_hands_recognized_but_one_or_separated_hands_are_not(self):
        left, right = self.hand(0.45), self.hand(0.55)
        self.assertTrue(joined_hands_from_landmarks([left, right], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, self.hand(0.83)], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, self.hand(0.55, upward=False)], 640, 480))

    def test_rotated_joined_hands_still_recognized(self):
        hands = [self.hand(0.45), self.hand(0.55)]
        rotated = [[SimpleNamespace(x=point.y, y=1 - point.x) for point in hand]
                   for hand in hands]
        self.assertTrue(joined_hands_from_landmarks(rotated, 480, 640))

    def test_recognizer_uses_two_hands_in_the_current_frame(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[self.hand(0.45), self.hand(0.55)])
        recognizer = app.MultiViewHandLandmarker(primary, alternate)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(recognizer.recognize_joined_hands(image, 100))
        self.assertEqual(recognizer.last_hand_count, 2)
        alternate.detect.assert_not_called()

    def test_hand_model_recovers_joined_hands_in_a_cropped_view(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[self.hand(0.45)])
        alternate.detect.side_effect = [
            SimpleNamespace(hand_landmarks=[]),
            SimpleNamespace(hand_landmarks=[self.hand(0.45), self.hand(0.55)]),
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
