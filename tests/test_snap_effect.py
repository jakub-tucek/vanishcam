import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from snap_effect import SNAP_SECONDS, SnapDetector, SnapRecognizer, render_snap, snap_anchor
import vanish_cam as app
from tests.test_video_output import LoopControlTests


def hand(pinched: bool):
    """21 landmarks of an upright hand: thumb pressed on the extended middle finger, or
    the middle fingertip flicked down to the palm with the thumb away from it."""
    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=0.8)    # Wrist.
    points[9] = SimpleNamespace(x=0.5, y=0.6)    # Middle knuckle: hand length 0.2.
    if pinched:
        points[12] = SimpleNamespace(x=0.5, y=0.5)
        points[4] = SimpleNamespace(x=0.51, y=0.51)
    else:
        points[12] = SimpleNamespace(x=0.5, y=0.7)
        points[4] = SimpleNamespace(x=0.6, y=0.55)
    return points


class SnapDetectorTests(unittest.TestCase):
    def feed(self, sequence):
        detector = SnapDetector()
        return [t for t, hands in sequence if detector.update(hands, 100, 100, t)]

    def test_pinch_then_flick_fires_once(self):
        fired = self.feed([(0.0, [hand(True)]), (0.033, [hand(True)]),
                           (0.066, [hand(False)]), (0.1, [hand(False)])])
        self.assertEqual(fired, [0.066])

    def test_slow_release_and_lone_flick_do_not_fire(self):
        self.assertEqual(self.feed([(0.0, [hand(True)]), (0.5, [hand(False)])]), [])
        self.assertEqual(self.feed([(0.0, [hand(False)]), (0.1, [hand(False)])]), [])
        self.assertEqual(self.feed([(0.0, [hand(True)]), (0.03, []), (0.06, [hand(True)])]), [])

    def test_cooldown_blocks_a_double_fire(self):
        sequence = [(0.0, [hand(True)]), (0.03, [hand(False)]),
                    (0.3, [hand(True)]), (0.33, [hand(False)]),
                    (1.0, [hand(True)]), (1.03, [hand(False)])]
        self.assertEqual(self.feed(sequence), [0.03, 1.03])

    def test_recognizer_reads_the_hand_landmarker(self):
        landmarker = Mock()
        recognizer = SnapRecognizer(landmarker)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((100, 100, 3), np.uint8))
        landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[hand(True)])
        self.assertFalse(recognizer.recognize_snap(image, 1, 0.0))
        self.assertIn("thumb on middle finger: yes", recognizer.describe())
        landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[hand(False)])
        self.assertTrue(recognizer.recognize_snap(image, 2, 0.05))
        landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[])
        recognizer.recognize_snap(image, 3, 0.1)
        self.assertEqual(recognizer.describe(), "No hand in view")


class SnapRenderTests(unittest.TestCase):
    def setUp(self):
        self.plate = np.full((120, 160, 3), 200, np.uint8)
        self.scene = self.plate.copy()
        self.cover = np.zeros((120, 160), np.float32)
        self.cover[30:110, 60:100] = 1
        self.scene[self.cover > 0] = 20
        self.anchor = snap_anchor(self.cover)

    def person_level(self, age, appearing):
        frame = render_snap(self.scene, self.plate, self.cover, self.anchor, age, appearing)
        return float(frame[100, 62].mean())  # Low on the body, where the smoke thins first.

    def test_person_hidden_after_vanish_and_shown_after_reappear(self):
        self.assertLess(self.person_level(0.0, appearing=False), 60)
        self.assertGreater(self.person_level(SNAP_SECONDS, appearing=False), 150)
        self.assertGreater(self.person_level(0.0, appearing=True), 150)
        self.assertLess(self.person_level(SNAP_SECONDS, appearing=True), 60)

    def test_anchor_covers_the_person(self):
        x, y, size = self.anchor
        self.assertEqual(x, 80)
        self.assertTrue(30 < y < 110)
        self.assertGreaterEqual(size, 60)


class SnapLoopTests(unittest.TestCase):
    run_loop = LoopControlTests.run_loop

    def test_snaps_vanish_and_bring_back_with_smoke(self):
        recognizer = Mock()
        recognizer.recognize_snap.side_effect = [True, True, False]
        recognizer.describe.return_value = ""
        times = (0.1, 4.1, 4.3, 6.0, 6.1, 8.0, 8.1)
        _, messages = self.run_loop("snap", times, (0,) * 6 + (ord("q"),), recognizer=recognizer)
        self.assertEqual([m for m in messages if m.startswith("[Snap]")],
                         ["[Snap] Vanishing.", "[Snap] Reappearing."])
        self.assertEqual(recognizer.recognize_snap.call_count, 3)

    def test_v_and_r_work_with_gestures_off(self):
        recognizer = Mock()
        times = (0.1, 4.1, 4.3, 6.0, 6.1, 6.2)
        keys = (ord("g"), ord("v"), 0, ord("r"), 0, ord("q"))
        _, messages = self.run_loop("snap", times, keys, recognizer=recognizer)
        self.assertEqual([m for m in messages if m.startswith("[Snap]")],
                         ["[Snap] Vanishing.", "[Snap] Reappearing."])
        recognizer.recognize_snap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
