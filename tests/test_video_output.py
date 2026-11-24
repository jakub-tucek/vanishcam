import argparse
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

import vanish_cam as app


def arguments(**overrides):
    values = dict(
        camera=0, width=960, height=540, fps=30.0, fullscreen=False,
        no_mirror=True, virtual_camera=None, virtual_backend=None, virtual_device=None,
        gesture_debug=False, mode="normal",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class VirtualCameraTests(unittest.TestCase):
    def test_bgr_format_and_actual_frame_dimensions(self):
        camera = Mock(device="OBS Virtual Camera")
        module = SimpleNamespace(Camera=Mock(return_value=camera), PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIs(app.open_virtual_camera(arguments(), 640, 480), camera)
        module.Camera.assert_called_once_with(
            width=640, height=480, fps=30.0, fmt="BGR", backend=None, device=None,
        )

    def test_unavailable_driver_auto_falls_back_but_explicit_mode_fails(self):
        module = SimpleNamespace(Camera=Mock(side_effect=RuntimeError("driver missing")),
                                 PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIsNone(app.open_virtual_camera(arguments(), 640, 480))
            for options in (dict(virtual_camera=True), dict(virtual_backend="obs")):
                with self.assertRaisesRegex(RuntimeError, "driver missing"):
                    app.open_virtual_camera(arguments(**options), 640, 480)

    def test_disabled_camera_does_not_open_device(self):
        module = SimpleNamespace(Camera=Mock())
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIsNone(app.open_virtual_camera(arguments(virtual_camera=False), 640, 480))
        module.Camera.assert_not_called()

    def test_send_preserves_colors_and_handles_noncontiguous_frames(self):
        camera = Mock(width=40, height=24)
        frame = np.zeros((24, 40, 3), np.uint8)
        frame[:] = (10, 70, 200)  # BGR: do not swap red and blue.
        app.send_virtual_frame(camera, frame[:, ::-1])
        sent = camera.send.call_args.args[0]
        self.assertTrue(sent.flags.c_contiguous)
        np.testing.assert_array_equal(sent, frame)

    def test_send_resizes_if_input_resolution_changes(self):
        camera = Mock(width=40, height=24)
        app.send_virtual_frame(camera, np.full((48, 80, 3), 42, np.uint8))
        sent = camera.send.call_args.args[0]
        self.assertEqual(sent.shape, (24, 40, 3))
        self.assertTrue((sent == 42).all())


class LauncherTests(unittest.TestCase):
    def launch(self, keys, cameras, mode="normal", camera_index=0):
        with patch.object(app, "open_camera", side_effect=cameras) as opening, \
             patch.object(app.cv2, "namedWindow"), \
             patch.object(app.cv2, "setMouseCallback"), \
             patch.object(app.cv2, "imshow"), \
             patch.object(app.cv2, "destroyWindow"), \
             patch.object(app.cv2, "getWindowProperty", return_value=1), \
             patch.object(app.cv2, "waitKey", side_effect=keys):
            result = app.choose_launch(app.ALL_MODES, mode, camera_index)
        return result, opening

    @staticmethod
    def camera(*reads):
        camera = Mock()
        camera.read.side_effect = reads or None
        if not reads:
            camera.read.return_value = (True, np.zeros((24, 40, 3), np.uint8))
        return camera

    def test_keys_pick_effect_and_switch_camera_preview(self):
        cameras = [self.camera(), self.camera()]
        result, opening = self.launch((ord("4"), ord("d"), 13), cameras)
        self.assertEqual(result, ("goku", 1))
        opening.assert_has_calls((call(0, 640, 360), call(1, 640, 360)))
        for camera in cameras:
            camera.release.assert_called_once()

    def test_every_effect_has_a_number_key(self):
        for index, mode in enumerate(app.ALL_MODES, 1):
            result, _ = self.launch((ord(str(index)), 13), [self.camera()])
            self.assertEqual(result, (mode, 0))

    def test_starts_with_remembered_choice(self):
        result, opening = self.launch((13,), [self.camera()], mode="itachi", camera_index=2)
        self.assertEqual(result, ("itachi", 2))
        opening.assert_called_once_with(2, 640, 360)

    def test_waits_for_an_available_device(self):
        camera = self.camera()
        result, _ = self.launch((13, ord("d"), 13), (RuntimeError("missing"), camera))
        self.assertEqual(result, ("normal", 1))
        camera.release.assert_called_once()

    def test_waits_for_first_preview_frame(self):
        camera = self.camera((False, None), (True, np.zeros((24, 40, 3), np.uint8)))
        result, _ = self.launch((13, 13), [camera])
        self.assertEqual(result, ("normal", 0))
        self.assertEqual(camera.read.call_count, 2)

    def test_quit_returns_nothing(self):
        result, _ = self.launch((ord("q"),), [self.camera()])
        self.assertIsNone(result)

    def test_launcher_text_fits_its_boxes(self):
        layout = app._launch_layout(len(app.ALL_MODES))
        for left, _, right, _ in layout["cards"]:
            for title, detail in app.MODE_LABELS.values():
                for text, scale, thickness in ((f"5  {title.upper()}", 0.7, 2), (detail, 0.47, 1)):
                    (width, _), _ = app.cv2.getTextSize(text, app.cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                    self.assertLessEqual(left + 18 + width, right, text)


class RunTests(unittest.TestCase):
    def test_mode_and_camera_on_command_line_skip_the_launcher(self):
        with patch.object(app, "parse_args", return_value=arguments(mode="naruto")), \
             patch.object(app, "choose_launch") as launcher, \
             patch.object(app, "run_camera", return_value=0) as camera:
            self.assertEqual(app.run(), 0)
        launcher.assert_not_called()
        self.assertEqual(camera.call_args.args[2], "naruto")

    def test_launcher_choice_is_used_and_remembered(self):
        args = arguments(mode=None, camera=None)
        with patch.object(app, "parse_args", return_value=args), \
             patch.object(app, "load_settings", return_value={"mode": "particles", "camera": 3}), \
             patch.object(app, "save_settings") as save, \
             patch.object(app, "choose_launch", return_value=("goku", 2)) as launcher, \
             patch.object(app, "run_camera", return_value=0) as camera:
            self.assertEqual(app.run(), 0)
        launcher.assert_called_once_with(app.ALL_MODES, "particles", 3, mirror=False)
        save.assert_called_once_with(mode="goku", camera=2)
        self.assertEqual(camera.call_args.args[0].camera, 2)


class MirrorTests(unittest.TestCase):
    def test_only_the_preview_is_mirrored(self):
        frame = np.zeros((24, 40, 3), np.uint8)
        frame[:, :10] = 200  # Bright stripe on the camera's left side.
        capture = Mock()
        capture.read.return_value = (True, frame)
        camera = Mock(width=40, height=24)
        shown = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(Mock(), Mock())))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=(0, 0, 0.1, 0.2)))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, image: shown.append(image)))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=(0, ord("q"))))
            mocks.enter_context(patch("builtins.print"))
            with ExitStack() as resources:
                app.run_camera(arguments(no_mirror=False), resources, "normal")
        np.testing.assert_array_equal(camera.send.call_args.args[0], frame)
        np.testing.assert_array_equal(shown[-1], frame[:, ::-1])


class HiddenAreaTests(unittest.TestCase):
    def test_mask_is_solid_inside_and_feathered_only_outside(self):
        mask = np.zeros((360, 640), np.float32)
        mask[100:260, 250:390] = 0.6
        widened = app.widen_mask(mask)
        self.assertTrue((widened[100:260, 250:390] > 0.97).all())
        self.assertGreater(widened[180, 245], 0.9)
        self.assertEqual(widened[10, 10], 0.0)

    def test_background_follows_camera_exposure(self):
        background = np.full((80, 120, 3), 100, np.uint8)
        frame = np.full((80, 120, 3), 120, np.uint8)
        mask = np.zeros((80, 120), np.float32)
        mask[20:60, 40:80] = 1.0
        frame[20:60, 40:80] = 255  # The person must not influence the gain.
        gain = app.match_exposure(background, frame, mask, None)
        np.testing.assert_allclose(gain, 1.2, atol=0.01)
        np.testing.assert_array_equal(app.apply_gain(background, gain)[0, 0], (120, 120, 120))


class HudTests(unittest.TestCase):
    def test_hud_changes_preview_only(self):
        frame = np.full((540, 960, 3), 90, np.uint8)
        shown, targets = app.draw_hud(frame, "normal", app.ALL_MODES, "Ready", "No hand in view",
                                      "Live: OBS Virtual Camera", app.HUD_LIVE, True)
        self.assertTrue(np.any(shown != frame))
        self.assertTrue((frame == 90).all())
        self.assertEqual(len(targets), len(app.ALL_MODES) + len(app.HUD_HELP_ITEMS))

    def test_clicks_map_to_effect_chips_and_key_buttons(self):
        frame = np.zeros((540, 960, 3), np.uint8)
        _, targets = app.draw_hud(frame, "goku", app.ALL_MODES, "", "", "Preview only",
                                  app.HUD_IDLE, True)
        keys = {key for _, key in targets}
        self.assertIn(ord("5"), keys)
        self.assertNotIn(ord("v"), keys)  # Goku vanishes by gesture only.
        (left, top, right, bottom), key = targets[2]
        self.assertEqual(app.hud_key_at(targets, (left + right) // 2, (top + bottom) // 2), ord("3"))
        self.assertIsNone(app.hud_key_at(targets, 480, 270))

    def test_status_follows_phase(self):
        self.assertIn("3s", app.hud_status("normal", "calibrating", 3, 1.0, None))
        self.assertEqual(app.hud_status("naruto", "ready", None, 1.0, None), app.GESTURE_HINTS["naruto"])
        self.assertIn("R to reappear", app.hud_status("normal", "vanished", None, 1.0, None))
        self.assertIn("0.5s", app.hud_status("goku", "ready", None, 1.0, 0.5))


class VideoLoopTests(unittest.TestCase):
    def run_mock_video(self, send_error=None, gesture=("No_Hand", 0.0), mode="normal"):
        capture = Mock()
        frames = [np.full((24, 40, 3), color, np.uint8) for color in (10, 40, 80, 120, 160)]
        capture.read.side_effect = [(True, frame) for frame in frames]
        segmenter, recognizer = Mock(), Mock()
        camera = Mock(width=40, height=24)
        camera.send.side_effect = send_error
        displayed = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(mode=mode)))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            opening = mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            segment = mocks.enter_context(patch.object(
                app, "person_mask", return_value=np.ones((24, 40), np.float32),
            ))
            mocks.enter_context(patch.object(app, "recognize_gesture", return_value=gesture))
            times = (0, 0, 0.1, 4.1, 4.3, 4.35, 5.0, 5.02, 8.2, 8.23)
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=times))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            destroy = mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: displayed.append(frame.copy())))
            keys = (0, 0, ord("v") if gesture[0] != "Victory" else 0, 0, ord("q"))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
            mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("Overlay vietato")))
            mocks.enter_context(patch("builtins.print"))
            if send_error is None:
                self.assertEqual(app.run(), 0)
            else:
                with self.assertRaisesRegex(RuntimeError, "send failed"):
                    app.run()
            opening.assert_called_once()
            self.assertEqual(opening.call_args.args[1:], (40, 24))
            capture.release.assert_called_once()
            segmenter.close.assert_called_once()
            recognizer.close.assert_called_once()
            camera.close.assert_called_once()
            destroy.assert_called_once()
            if mode == "particles":
                segment.assert_called_once()  # A single silhouette capture.
        return frames, camera, displayed

    def test_preview_and_virtual_output_are_clean_in_all_phases(self):
        frames, camera, displayed = self.run_mock_video()
        self.assertEqual(len(displayed), 5)
        for sent, preview in zip(camera.send.call_args_list, displayed):
            np.testing.assert_array_equal(sent.args[0], preview)
        np.testing.assert_array_equal(displayed[0], frames[0])  # Calibrazione.
        np.testing.assert_array_equal(displayed[2], frames[2])  # Pronto.
        np.testing.assert_array_equal(displayed[4], frames[1])  # Vanished: background only.

    def test_first_gesture_starts_dissolve_in_preview_and_virtual_output(self):
        frames, camera, displayed = self.run_mock_video(gesture=("Victory", 0.9))
        np.testing.assert_array_equal(camera.send.call_args_list[2].args[0], displayed[2])
        self.assertTrue(np.any(displayed[3] != frames[3]))  # Animazione in corso.
        self.assertTrue(np.any(displayed[3] != frames[1]))  # Non ancora sparito.
        np.testing.assert_array_equal(displayed[4], frames[1])  # Animazione completata.

    def test_particles_use_same_v_gesture_and_virtual_output(self):
        frames, camera, displayed = self.run_mock_video(
            gesture=("Victory", 0.9), mode="particles",
        )
        self.assertTrue(np.any(displayed[2] != frames[2]))  # Parte subito.
        self.assertTrue(np.any(displayed[3] != frames[3]))  # Corpo si dissolve.
        np.testing.assert_array_equal(displayed[4], frames[1])
        for sent, preview in zip(camera.send.call_args_list, displayed):
            np.testing.assert_array_equal(sent.args[0], preview)

    def test_all_devices_close_on_send_error(self):
        self.run_mock_video(send_error=RuntimeError("send failed"))

    def test_goku_vanishes_and_reappears_only_after_one_second_holds(self):
        first_hold = tuple(4.3 + index * 0.101 for index in range(11))
        second_hold = tuple(6.4 + index * 0.101 for index in range(11))
        times = (0.1, 4.1) + first_hold + (5.4, 5.64, 5.8, 6.2) \
            + second_hold + (7.5, 7.74, 7.9)
        clock = [0.0]
        frames = [np.full((80, 120, 3), 40 if i < 2 else 160, np.uint8)
                  for i in range(len(times))]
        capture = Mock()
        cursor = [0]

        def read_frame():
            index = cursor[0]
            cursor[0] += 1
            clock[0] = times[index]
            return True, frames[index]

        capture.read.side_effect = read_frame
        segmenter = Mock()
        recognizer = Mock()
        recognizer.recognize_pose.side_effect = (
            [True] * 11 + [False] * 2 + [True] * 11 + [False]
        )
        camera = Mock(width=120, height=80)
        sound = Mock()
        mask = np.zeros((80, 120), np.float32)
        mask[10:70, 30:90] = 1.0
        displayed = []
        keys = [0] * len(times)
        keys[2] = ord("v")  # Manual activation is forbidden in Goku mode.
        keys[14] = ord("r")  # Manual return is forbidden too.
        keys[-1] = ord("q")
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            masks = mocks.enter_context(patch.object(app, "person_mask", return_value=mask))
            sound_factory = mocks.enter_context(patch.object(app, "VanishSound", return_value=sound))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=lambda: clock[0]))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: displayed.append(frame.copy())))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
            printed = mocks.enter_context(patch("builtins.print"))
            with ExitStack() as resources:
                self.assertEqual(app.run_camera(arguments(mode="goku", goku_delay=1.0), resources, "goku"), 0)

        # 2 teleport masks plus live masking while vanished (LIVE_VANISH_MODES).
        self.assertEqual(masks.call_count, 14)
        sound_factory.assert_called_once_with(app.TELEPORT_SOUND_PATH)
        self.assertEqual(sound.play.call_count, 2)
        countdown = [call.args[0] for call in printed.call_args_list
                     if call.args and call.args[0].startswith("[Goku] ")]
        self.assertEqual(countdown, [
            "[Goku] Disappearance in 1...",
            "[Goku] Disappearance!",
            "[Goku] Reappearance in 1...",
            "[Goku] Reappearance!",
        ])
        np.testing.assert_array_equal(displayed[2], frames[2])
        np.testing.assert_array_equal(displayed[14], frames[1])
        # Hidden area shows the exposure-matched background (gain capped at 1.33), never the live person.
        expected = (frames[1][40, 60].astype(np.float32) * np.float32(1.33)).astype(np.uint8)
        np.testing.assert_array_equal(displayed[16][40, 60], expected)
        np.testing.assert_allclose(displayed[16][0, 0], frames[16][0, 0], atol=3)
        np.testing.assert_array_equal(displayed[29][40, 60], frames[29][40, 60])
        for sent, preview in zip(camera.send.call_args_list, displayed):
            np.testing.assert_array_equal(sent.args[0], preview)


if __name__ == "__main__":
    unittest.main()
