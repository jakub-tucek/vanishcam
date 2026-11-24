import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import cv2
import numpy as np

from itachi_effect import (
    BODY_RELEASE_SECONDS, EFFECT_SECONDS, ItachiCrowEffect, ItachiPoseRecognizer,
    analyze_arm_cross, load_sharingan_art,
)
import vanish_cam as app


def pose(left_wrist_x, right_wrist_x, wrist_y=0.70, visibility=0.8,
         left_shoulder_x=0.62, right_shoulder_x=0.38):
    """33 pose landmarks; unmirrored camera, so the person's left shoulder is on the right."""
    points = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
    points[11] = SimpleNamespace(x=left_shoulder_x, y=0.55, visibility=0.95)
    points[12] = SimpleNamespace(x=right_shoulder_x, y=0.55, visibility=0.95)
    points[15] = SimpleNamespace(x=left_wrist_x, y=wrist_y, visibility=visibility)
    points[16] = SimpleNamespace(x=right_wrist_x, y=wrist_y, visibility=visibility)
    return points


def scene():
    background = np.full((240, 320, 3), (96, 157, 191), np.uint8)
    person = background.copy()
    mask = np.zeros((240, 320), np.float32)
    cv2.ellipse(mask, (160, 115), (60, 105), 0, 0, 360, 1, -1)
    person[mask > .5] = (145, 95, 55)
    return person, background, mask


class ItachiGestureTests(unittest.TestCase):
    def test_crossed_arms_at_chest_trigger(self):
        cross = analyze_arm_cross(pose(0.45, 0.55))
        self.assertTrue(cross.ok, cross)
        self.assertLess(cross.measures["separation"], 0)

    def test_hands_meeting_without_crossing_still_count_as_crossed_wrists(self):
        self.assertTrue(analyze_arm_cross(pose(0.52, 0.48)).crossed)

    def test_poses_that_must_not_trigger(self):
        cases = {
            "own side": (pose(0.62, 0.38), "crossed"),
            "resting low": (pose(0.45, 0.55, wrist_y=0.95), "at_chest"),
            "wrists out of view": (pose(0.45, 0.55, visibility=0.2), "visible"),
        }
        for name, (landmarks, failing) in cases.items():
            with self.subTest(name):
                cross = analyze_arm_cross(landmarks)
                self.assertFalse(cross.ok)
                self.assertFalse(getattr(cross, failing))
        self.assertFalse(analyze_arm_cross(pose(0.45, 0.55, left_shoulder_x=0.51,
                                                right_shoulder_x=0.49)).ok)  # Side-on.
        self.assertFalse(analyze_arm_cross(None).body)

    def test_feedback_names_what_is_missing(self):
        self.assertIn("Arms crossed: no", analyze_arm_cross(pose(0.62, 0.38)).describe())
        self.assertEqual(analyze_arm_cross(None).describe(), "No body in view")

    def test_recognizer_reads_the_pose_landmarker(self):
        landmarker = Mock()
        landmarker.detect_for_video.return_value = SimpleNamespace(pose_landmarks=[pose(0.45, 0.55)])
        recognizer = ItachiPoseRecognizer(landmarker)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((240, 320, 3), np.uint8))
        self.assertTrue(recognizer.recognize_pose(image, 100))
        self.assertTrue(recognizer.last_cross.ok)
        landmarker.detect_for_video.return_value = SimpleNamespace(pose_landmarks=[])
        self.assertFalse(recognizer.recognize_pose(image, 200))


class ItachiAnimationTests(unittest.TestCase):
    def test_local_body_removal_crows_flap_and_final_frame_is_clean(self):
        person, background, mask = scene()
        effect = ItachiCrowEffect(person, mask)
        self.assertEqual(len(effect.crows), 78)
        self.assertEqual(sum(crow.hero for crow in effect.crows), 4)
        self.assertGreaterEqual(len(effect.feathers), 30)
        self.assertGreater(float(np.percentile(effect.release_map[mask > .5], 75)), 1.8)
        self.assertGreaterEqual(float(effect.release_map[75, 160]), 1.86)
        self.assertLessEqual(float(effect.release_map[35, 160]), 1.71)
        first = effect.render(background, 0)
        middle = effect.render(background, BODY_RELEASE_SECONDS * .55)
        later = effect.render(background, BODY_RELEASE_SECONDS)
        final = effect.render(background, EFFECT_SECONDS)
        self.assertTrue(np.array_equal(first, person))
        self.assertGreater(np.count_nonzero(np.any(middle != person, axis=2)), 100)
        self.assertFalse(np.array_equal(middle, later))
        self.assertFalse(np.array_equal(final, background))
        self.assertLess(int(final[0, 0, 0]), int(background[0, 0, 0]))
        self.assertTrue(np.array_equal(final, effect.render(background, EFFECT_SECONDS + 1)))
        self.assertFalse(np.array_equal(effect.sprites[0], effect.sprites[2]))

    def test_sharingan_appears_briefly_without_a_black_rectangle(self):
        person, background, mask = scene()
        art = load_sharingan_art()
        self.assertEqual(art.shape[2], 4)
        self.assertEqual(int(art[0, 0, 3]), 0)
        self.assertTrue(np.all(art[:, 0, 3] == 0))
        self.assertTrue(np.all(art[:, -1, 3] == 0))
        effect = ItachiCrowEffect(person, mask)
        with_eyes = effect.render(background, 1.0)
        with patch.object(effect, '_draw_sharingan'):
            without_eyes = effect.render(background, 1.0)
        self.assertFalse(np.array_equal(with_eyes[:90], without_eyes[:90]))
        late = effect.render(background, 3.0)
        with patch.object(effect, '_draw_sharingan'):
            late_without_eyes = effect.render(background, 3.0)
        self.assertTrue(np.array_equal(late, late_without_eyes))

    def test_reverse_animation_reconstructs_person(self):
        person, background, mask = scene()
        effect = ItachiCrowEffect(person, mask)
        self.assertTrue(np.array_equal(effect.render(background, 0, appearing=True),
                                       effect.render(background, EFFECT_SECONDS)))
        self.assertTrue(np.array_equal(effect.render(background, EFFECT_SECONDS, appearing=True), person))


if __name__ == "__main__":
    unittest.main()
