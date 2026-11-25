import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from clone_effect import CLONE_SECONDS, CloneSealRecognizer, CloneSquad, analyze_clone_seal
import vanish_cam as app
from tests import test_video_output


def hand(wrist, direction, folded=True, length=0.1):
    """21 landmarks; index and middle point along direction, ring and pinky folded or open."""
    wrist, along = np.array(wrist, float), np.array(direction, float)
    along /= np.linalg.norm(along)
    side = np.array([-along[1], along[0]])
    points = [wrist.copy() for _ in range(21)]
    for knuckle, tip, offset in ((5, 8, 0.15), (9, 12, 0.0), (13, 16, -0.15), (17, 20, -0.3)):
        base = wrist + along * length + side * offset * length
        points[knuckle] = base
        bend = -0.1 if folded and knuckle >= 13 else 0.85
        points[tip] = base + along * bend * length
    return [SimpleNamespace(x=float(x), y=float(y)) for x, y in points]


CENTER = np.array([0.5, 0.45])


def seal():
    """Horizontal fingers of one hand crossing the upright fingers of the other at CENTER."""
    return [hand(CENTER - np.array([0.14, 0]), (1, 0)), hand(CENTER + np.array([0, 0.14]), (0, -1))]


class CloneSealTests(unittest.TestCase):
    def test_crossed_fingers_are_the_seal(self):
        result = analyze_clone_seal(seal(), 1, 1)
        self.assertTrue(result.ok, result)
        self.assertEqual(result.describe(), "Two fingers out: yes   crossed: yes")

    def test_look_alikes_are_not(self):
        cases = {
            "palms together": [hand((0.47, 0.6), (0, -1)), hand((0.53, 0.6), (0, -1))],
            "open hands crossed": [hand(CENTER - np.array([0.14, 0]), (1, 0), folded=False),
                                   hand(CENTER + np.array([0, 0.14]), (0, -1), folded=False)],
            "crossed but apart": [hand((0.1, 0.45), (1, 0)), hand((0.8, 0.6), (0, -1))],
        }
        for name, hands in cases.items():
            with self.subTest(name):
                self.assertFalse(analyze_clone_seal(hands, 1, 1).ok)
        self.assertEqual(analyze_clone_seal(seal()[:1], 1, 1).describe(), "Show both hands")
        self.assertEqual(analyze_clone_seal([], 1, 1).describe(), "No hands in view")

    def test_recognizer_reads_the_hand_landmarker(self):
        landmarker = Mock()
        landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=seal())
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((100, 100, 3), np.uint8))
        recognizer = CloneSealRecognizer(landmarker)
        self.assertTrue(recognizer.recognize_seal(image, 1))
        self.assertTrue(recognizer.last_seal.ok)


class CloneRenderTests(unittest.TestCase):
    def setUp(self):
        self.scene = np.zeros((120, 200, 3), np.uint8)
        self.mask = np.zeros((120, 200), np.float32)
        self.mask[30:120, 90:110] = 1
        self.scene[self.mask > 0] = 255
        self.squad = CloneSquad(self.mask)

    def bright_columns(self, frame):
        return np.flatnonzero(frame[100].max(axis=1) > 200)

    def test_steady_clones_stand_beside_the_person(self):
        columns = self.bright_columns(self.squad.render(self.scene, self.mask))
        self.assertLess(columns.min(), 60)
        self.assertGreater(columns.max(), 140)
        self.assertTrue(((columns >= 90) & (columns < 110)).any())

    def test_clones_pop_in_and_out(self):
        for phase, age, expect in (("appearing", 0.0, False), ("appearing", CLONE_SECONDS, True),
                                   ("leaving", CLONE_SECONDS, False)):
            with self.subTest(phase=phase, age=age):
                frame = self.squad.render(self.scene, self.mask, age, phase)
                clone = frame[100, 30:60].max() > 200
                self.assertEqual(clone, expect)

    def test_nobody_to_clone(self):
        with self.assertRaises(ValueError):
            CloneSquad(np.zeros((120, 200), np.float32))


class CloneLoopTests(unittest.TestCase):
    run_loop = test_video_output.LoopControlTests.run_loop

    def test_seal_summons_and_dismisses(self):
        answers = iter([True, True, True, False, False, True, True, True])
        recognizer = Mock()
        recognizer.recognize_seal.side_effect = lambda image, ts: next(answers, False)
        recognizer.last_seal.describe.return_value = ""
        times = (0.1, 4.1, 4.2, 4.4, 4.55, 6.5, 6.6, 7.0, 7.1, 7.3, 7.45, 9.4, 9.5)
        _, messages = self.run_loop("clones", times, (0,) * 12 + (ord("q"),), recognizer=recognizer)
        self.assertEqual([m for m in messages if m.startswith("[Clones]")],
                         ["[Clones] Summoned.", "[Clones] Dismissed."])

    def test_v_and_r_with_gestures_off(self):
        recognizer = Mock()
        times = (0.1, 4.1, 4.3, 6.5, 6.6, 6.7)
        keys = (ord("g"), ord("v"), 0, ord("r"), 0, ord("q"))
        _, messages = self.run_loop("clones", times, keys, recognizer=recognizer)
        self.assertEqual([m for m in messages if m.startswith("[Clones]")],
                         ["[Clones] Summoned.", "[Clones] Dismissed."])
        recognizer.recognize_seal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
