import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from lava_effect import LAVA_SECONDS, LavaEffect, ThumbsUpRecognizer
import vanish_cam as app
from tests import test_video_output


def gesture_result(score, hands=1):
    return SimpleNamespace(hand_landmarks=[[]] * hands,
                           gestures=[[SimpleNamespace(category_name="Thumb_Up", score=score)]] * hands)


class ThumbsUpTests(unittest.TestCase):
    def test_score_threshold_and_feedback(self):
        model = Mock()
        recognizer = ThumbsUpRecognizer(model)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB, data=np.zeros((10, 10, 3), np.uint8))
        for score, expected in ((0.72, True), (0.4, False)):
            model.recognize_for_video.return_value = gesture_result(score)
            self.assertEqual(recognizer.recognize_thumb(image, 1), expected)
        self.assertIn("Thumbs up: 40%", recognizer.describe())
        model.recognize_for_video.return_value = gesture_result(0.0, hands=0)
        self.assertFalse(recognizer.recognize_thumb(image, 2))
        self.assertEqual(recognizer.describe(), "No hand in view")


class LavaRenderTests(unittest.TestCase):
    def setUp(self):
        self.background = np.zeros((120, 200, 3), np.uint8)
        self.frame = self.background.copy()
        self.mask = np.zeros((120, 200), np.float32)
        self.mask[20:96, 90:110] = 1
        self.frame[self.mask > 0] = (0, 255, 0)
        self.effect = LavaEffect(self.frame, self.mask)

    def person_pixels(self, image):
        green = (image[:, :, 1] > 150) & (image[:, :, 0] < 80) & (image[:, :, 2] < 80)
        return int(green.sum())

    def test_sinks_out_of_view_and_rises_back(self):
        start = self.person_pixels(self.effect.render(self.background, 0.0))
        self.assertGreater(start, 1000)
        self.assertEqual(self.person_pixels(self.effect.render(self.background, LAVA_SECONDS)), 0)
        self.assertEqual(self.person_pixels(self.effect.render(self.background, 0.0, appearing=True)), 0)
        self.assertGreater(self.person_pixels(self.effect.render(self.background, LAVA_SECONDS, True)),
                           start * 0.9)

    def test_pool_shows_mid_effect_and_drains_after(self):
        middle = self.effect.render(self.background, 2.0)
        self.assertGreater(middle[110, :, 2].mean(), 100)       # Glowing red/orange at the bottom.
        drained = self.effect.render(self.background, LAVA_SECONDS)
        self.assertLess(drained[110, :, 2].mean(), 5)

    def test_nobody_to_sink(self):
        with self.assertRaises(ValueError):
            LavaEffect(self.frame, np.zeros_like(self.mask))


class LavaLoopTests(unittest.TestCase):
    run_loop = test_video_output.LoopControlTests.run_loop

    def test_thumbs_up_sinks_and_rises(self):
        answers = iter([True, True, True, False, False, True, True, True])
        recognizer = Mock()
        recognizer.recognize_thumb.side_effect = lambda image, ts: next(answers, False)
        recognizer.describe.return_value = ""
        # Hold 0.3 s, sink, release 0.4 s (the gate re-arms after 0.35 s), hold again, rise.
        times = (0.1, 4.1, 4.2, 4.4, 4.55, 9.0, 9.1, 9.5, 9.6, 9.8, 9.95, 14.4, 14.5)
        _, messages = self.run_loop("lava", times, (0,) * 12 + (ord("q"),), recognizer=recognizer)
        self.assertEqual([m for m in messages if m.startswith("[Lava]")],
                         ["[Lava] Sinking into the lava.", "[Lava] Rising from the lava."])

    def test_v_and_r_with_gestures_off(self):
        recognizer = Mock()
        times = (0.1, 4.1, 4.3, 9.0, 9.1, 9.2)
        keys = (ord("g"), ord("v"), 0, ord("r"), 0, ord("q"))
        _, messages = self.run_loop("lava", times, keys, recognizer=recognizer)
        self.assertEqual([m for m in messages if m.startswith("[Lava]")],
                         ["[Lava] Sinking into the lava.", "[Lava] Rising from the lava."])
        recognizer.recognize_thumb.assert_not_called()


if __name__ == "__main__":
    unittest.main()
