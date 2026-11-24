import argparse
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def test_missing_driver_is_reported_without_retrying(self):
        module = SimpleNamespace(
            Camera=Mock(side_effect=RuntimeError("'obs' backend: OBS Virtual Camera is not installed")),
            PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module):
            with self.assertRaisesRegex(app.VirtualCameraMissing, "No OBS Virtual Camera driver"):
                app.open_virtual_camera(arguments(), 640, 480)
            with ExitStack() as resources, patch("builtins.print"):
                outputs = app.VideoOutputs(arguments(), resources)
                frame = np.zeros((24, 40, 3), np.uint8)
                outputs.send(frame, 0.0)
                outputs.send(frame, 10.0)
        self.assertEqual(module.Camera.call_count, 2)  # The direct call plus one attempt, no retries.
        self.assertEqual(outputs.badge(), ("No virtual camera driver (see README)", app.HUD_WARN))

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


class CameraSwitcherTests(unittest.TestCase):
    def wait(self, switcher):
        for _ in range(200):
            result = switcher.poll()
            if result is not None:
                return result
            time.sleep(0.01)
        raise AssertionError("camera search never finished")

    @staticmethod
    def camera(works=True):
        camera = Mock()
        camera.read.return_value = (works, np.zeros((24, 40, 3), np.uint8) if works else None)
        return camera

    def test_skips_missing_and_dead_cameras(self):
        dead, alive = self.camera(works=False), self.camera()
        opened = {1: RuntimeError("missing"), 2: dead, 3: alive}

        def open_camera(index, width, height):
            result = opened.get(index, RuntimeError("missing"))
            if isinstance(result, Exception):
                raise result
            return result

        with patch.object(app, "open_camera", side_effect=open_camera) as opening:
            switcher = app.CameraSwitcher(640, 360)
            switcher.start(0)
            self.assertEqual(self.wait(switcher), (3, alive))
        self.assertEqual([c.args[0] for c in opening.call_args_list], [1, 2, 3])
        dead.release.assert_called_once()
        alive.release.assert_not_called()

    def test_wraps_around_and_reports_when_nothing_else_works(self):
        with patch.object(app, "open_camera", side_effect=RuntimeError("missing")) as opening:
            switcher = app.CameraSwitcher(640, 360)
            switcher.start(8)
            self.assertEqual(self.wait(switcher), (None, None))
        self.assertEqual([c.args[0] for c in opening.call_args_list], [9, 0, 1, 2, 3, 4, 5, 6, 7])

    def test_known_names_skip_virtual_cameras_and_missing_indexes(self):
        names = ["MacBook Pro Camera", "OBS Virtual Camera", "Studio Display Camera"]
        with patch.object(app, "open_camera", return_value=self.camera()) as opening:
            switcher = app.CameraSwitcher(640, 360, names)
            switcher.start(0)
            self.assertEqual(self.wait(switcher)[0], 2)
            switcher.start(2)
            self.assertEqual(self.wait(switcher)[0], 0)
        self.assertEqual([c.args[0] for c in opening.call_args_list], [2, 0])
        self.assertEqual(switcher.label(2), "Studio Display Camera")

    def test_only_camera_reports_nothing_else(self):
        with patch.object(app, "open_camera") as opening:
            switcher = app.CameraSwitcher(640, 360, ["MacBook Pro Camera", "OBS Virtual Camera"])
            switcher.start(0)
            self.assertEqual(self.wait(switcher), (None, None))
        opening.assert_not_called()

    def test_slow_open_does_not_block_the_caller(self):
        def slow_open(index, width, height):
            time.sleep(0.3)
            return self.camera()

        with patch.object(app, "open_camera", side_effect=slow_open):
            switcher = app.CameraSwitcher(640, 360)
            started = time.perf_counter()
            switcher.start(0)
            self.assertLess(time.perf_counter() - started, 0.05)
            self.assertTrue(switcher.searching)
            self.assertIsNone(switcher.poll())
            self.assertEqual(self.wait(switcher)[0], 1)


class RunTests(unittest.TestCase):
    def test_command_line_effect_and_camera_skip_saved_settings(self):
        with patch.object(app, "parse_args", return_value=arguments(mode="naruto")), \
             patch.object(app, "load_settings") as settings, \
             patch.object(app, "run_camera", return_value=0) as camera:
            self.assertEqual(app.run(), 0)
        settings.assert_not_called()
        self.assertEqual(camera.call_args.args[2], "naruto")

    def test_starts_live_with_remembered_effect_and_camera(self):
        args = arguments(mode=None, camera=None)
        with patch.object(app, "parse_args", return_value=args), \
             patch.object(app, "load_settings", return_value={"mode": "particles", "camera": 3}), \
             patch.object(app, "run_camera", return_value=0) as camera:
            self.assertEqual(app.run(), 0)
        self.assertEqual(camera.call_args.args[2], "particles")
        self.assertEqual(camera.call_args.args[0].camera, 3)

    def test_first_start_defaults_to_normal_on_camera_zero(self):
        args = arguments(mode=None, camera=None)
        with patch.object(app, "parse_args", return_value=args), \
             patch.object(app, "load_settings", return_value={"mode": "bogus", "camera": 42}), \
             patch.object(app, "run_camera", return_value=0) as camera:
            self.assertEqual(app.run(), 0)
        self.assertEqual(camera.call_args.args[2], "normal")
        self.assertEqual(camera.call_args.args[0].camera, 0)


class VideoOutputsTests(unittest.TestCase):
    def outputs(self, opened, **options):
        resources = ExitStack()
        self.addCleanup(resources.close)
        patcher = patch.object(app, "open_virtual_camera", side_effect=opened)
        self.opening = patcher.start()
        self.addCleanup(patcher.stop)
        return app.VideoOutputs(arguments(**options), resources)

    def test_busy_driver_is_retried_every_few_seconds_until_it_opens(self):
        camera = Mock(width=40, height=24, device="OBS Virtual Camera")
        outputs = self.outputs([None, camera])
        frame = np.zeros((24, 40, 3), np.uint8)
        outputs.send(frame, 0.0)
        outputs.send(frame, 1.0)  # Too early for a retry.
        self.assertEqual(self.opening.call_count, 1)
        self.assertEqual(outputs.badge(), ("Virtual camera busy (quit OBS?)", app.HUD_WARN))
        outputs.send(frame, 3.1)
        self.assertEqual(self.opening.call_count, 2)
        camera.send.assert_called_once()
        self.assertEqual(outputs.badge(), ("Live: OBS Virtual Camera", app.HUD_LIVE))

    def test_disabled_virtual_camera_is_not_retried(self):
        outputs = self.outputs([None], virtual_camera=False)
        frame = np.zeros((24, 40, 3), np.uint8)
        outputs.send(frame, 0.0)
        outputs.send(frame, 10.0)
        self.assertEqual(self.opening.call_count, 1)
        self.assertEqual(outputs.badge(), ("Preview only", app.HUD_IDLE))


class LoopControlTests(unittest.TestCase):
    """Drive run_camera frame by frame: each frame read advances the fake clock."""

    def run_loop(self, mode, times, keys, recognizer=None, person_mask=None, extra=(), **options):
        clock = [0.0]
        frames = iter(times)

        def read():
            clock[0] = next(frames)
            return True, np.full((80, 120, 3), 90, np.uint8)

        capture = Mock()
        capture.read.side_effect = read
        recognizer = recognizer or Mock()
        mask = np.zeros((80, 120), np.float32)
        mask[20:70, 40:80] = 1.0
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "save_settings"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            tasks = mocks.enter_context(patch.object(app, "create_tasks", return_value=(Mock(), recognizer)))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=None))
            mocks.enter_context(patch.object(app, "VanishSound"))
            mocks.enter_context(patch.object(app, "person_mask", side_effect=person_mask or (lambda *a: mask)))
            mocks.enter_context(patch.object(app, "recognize_gesture", return_value=("No_Hand", 0.0)))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=lambda: clock[0]))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows", "imshow"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
            for target in extra:
                mocks.enter_context(target)
            printed = mocks.enter_context(patch("builtins.print"))
            with ExitStack() as resources:
                self.assertEqual(app.run_camera(arguments(mode=mode, **options), resources, mode), 0)
        messages = [c.args[0] for c in printed.call_args_list if c.args]
        return tasks, messages

    def test_timestamps_stay_strictly_increasing_within_one_frame(self):
        stamps = []
        recognizer = Mock()
        recognizer.recognize_pose.return_value = False
        mask = np.ones((80, 120), np.float32)
        # With blur on, the segmenter runs for the background every frame; V in Itachi
        # runs it again in the same frame. MediaPipe rejects a repeated timestamp.
        self.run_loop(
            "itachi", (0.1, 4.1, 4.3, 4.5, 4.7), (0, 0, ord("v"), 0, ord("q")),
            recognizer=recognizer,
            person_mask=lambda segmenter, image, ts, w, h: stamps.append(ts) or mask,
            extra=(patch.object(app, "ItachiCrowEffect"),), background="blur",
        )
        self.assertGreaterEqual(len(stamps), 5)
        self.assertEqual(stamps, sorted(set(stamps)))

    def test_number_keys_switch_effect_and_load_its_models(self):
        recognizer = Mock()
        recognizer.recognize_joined_hands.return_value = False
        tasks, messages = self.run_loop(
            "normal", (0.1, 4.1, 4.3, 4.5, 4.7), (0, 0, ord("2"), 0, ord("q")), recognizer=recognizer)
        self.assertEqual([c.args[0] for c in tasks.call_args_list], ["normal", "naruto"])
        self.assertIn("[Webcam] Effect: Naruto", messages)

    def test_g_pauses_gesture_recognition_until_pressed_again(self):
        recognize = Mock(return_value=("Victory", 0.95))
        times = (0.1, 4.1, 4.3, 4.5, 4.7, 4.9, 5.1)
        keys = (0, ord("g"), 0, 0, ord("g"), 0, ord("q"))
        _, messages = self.run_loop("normal", times, keys,
                                    extra=(patch.object(app, "recognize_gesture", recognize),))
        self.assertIn("[Webcam] Gestures off: use V and R.", messages)
        self.assertIn("[Webcam] Gestures on.", messages)
        # Gestures went off at 4.1 and back on at 4.7: only the 4.9 frame was checked.
        self.assertEqual(recognize.call_count, 1)

    def test_goku_keys_work_only_while_gestures_are_off(self):
        recognizer = Mock()
        recognizer.recognize_pose.return_value = False
        teleport = patch.object(app, "GokuTeleport")
        _, messages = self.run_loop(
            "goku", (0.1, 4.1, 4.3, 4.5, 4.7), (0, ord("v"), ord("g"), ord("v"), ord("q")),
            recognizer=recognizer, extra=(teleport,))
        self.assertEqual([m for m in messages if m.startswith("[Goku]")], ["[Goku] Disappearance!"])

    def test_never_starts_on_its_own_virtual_camera(self):
        names = patch.object(app, "camera_names", return_value=["OBS Virtual Camera", "MacBook Pro Camera"])
        _, messages = self.run_loop("normal", (0.1, 0.2), (0, ord("q")), extra=(names,))
        self.assertTrue(any("using MacBook Pro Camera instead" in m for m in messages))

    def test_switching_during_background_capture_keeps_capturing(self):
        times = (0.1, 4.1, 4.3, 4.5) + tuple(4.9 + i * 0.4 for i in range(10))
        keys = [0] * len(times)
        keys[2] = ord(" ")
        keys[3] = ord("3")
        keys[-1] = ord("q")
        _, messages = self.run_loop("normal", times, keys)
        ready = [m for m in messages if m.startswith("[Webcam] Background ready")]
        self.assertEqual(len(ready), 2)


class BackgroundStyleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.frame = rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)
        self.mask = np.zeros((90, 160), np.float32)
        self.mask[20:80, 60:100] = 1.0

    def style(self, choice, images=()):
        style = app.BackgroundStyle(list(images), choice)
        style.update_mask(self.mask, 0.0)
        return style

    def test_small_detached_blobs_are_dropped_but_a_second_person_stays(self):
        mask = np.zeros((180, 320), np.float32)
        mask[40:180, 120:200] = 1.0   # The person.
        mask[10:25, 260:300] = 1.0    # A lamp misread as a person.
        mask[60:180, 10:70] = 1.0     # A second person, about half the size.
        cleaned = app.keep_person_regions(mask)
        self.assertEqual(cleaned[100, 160], 1.0)
        self.assertEqual(cleaned[17, 280], 0.0)
        self.assertEqual(cleaned[120, 40], 1.0)

    def test_edge_refinement_follows_the_image(self):
        frame = np.zeros((90, 160, 3), np.uint8)
        frame[:, 80:] = 255            # A hard edge at x=80.
        rough = np.zeros((90, 160), np.float32)
        rough[:, 70:] = 1.0            # The mask overshoots by 10 px.
        refined = app.refine_mask_edges(rough, frame)
        self.assertLess(refined[45, 72], rough[45, 72] - 0.3)
        self.assertGreater(refined[45, 120], 0.9)

    def test_off_returns_the_frame_untouched(self):
        style = self.style("off")
        self.assertIs(style.scene(self.frame), self.frame)
        self.assertFalse(style.mask_due(1.0))

    def test_blur_keeps_the_person_sharp_and_softens_the_room(self):
        scene = self.style("blur").scene(self.frame)
        np.testing.assert_array_equal(scene[40:60, 70:90], self.frame[40:60, 70:90])
        room = scene[:, :40].astype(float)
        self.assertLess(np.abs(np.diff(room, axis=1)).mean(),
                        np.abs(np.diff(self.frame[:, :40].astype(float), axis=1)).mean() / 3)

    def test_image_replaces_the_room_and_hides_everything_when_vanished(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "beach.png"
            app.cv2.imwrite(str(path), np.full((300, 300, 3), (200, 100, 50), np.uint8))
            style = self.style(str(path), [path])
            scene = style.scene(self.frame)
            np.testing.assert_array_equal(scene[5, 5], (200, 100, 50))
            np.testing.assert_array_equal(scene[50, 80], self.frame[50, 80])
            vanished = style.scene(self.frame, person_visible=False)
            np.testing.assert_array_equal(vanished[50, 80], (200, 100, 50))
            self.assertEqual(style.label, "beach")

    def test_plate_is_cached_per_background_and_style(self):
        style = self.style("blur")
        background = self.frame.copy()
        plate = style.plate(background)
        self.assertIs(style.plate(background), plate)
        self.assertIsNot(style.plate(background.copy()), plate)
        self.assertIsNone(style.plate(None))

    def test_cycle_goes_off_blur_then_each_image(self):
        style = app.BackgroundStyle([Path("a.png"), Path("b.jpg")], "off")
        labels = []
        for _ in range(4):
            style.cycle()
            labels.append(style.label)
        self.assertEqual(labels, ["blur", "a", "b", "off"])

    def test_cover_resize_fills_without_distortion(self):
        wide = np.zeros((100, 400, 3), np.uint8)
        wide[:, 150:250] = 255  # A centred white band survives the crop.
        result = app.cover_resize(wide, 160, 90)
        self.assertEqual(result.shape, (90, 160, 3))
        self.assertEqual(result[45, 80, 0], 255)

    def test_setup_resolves_files_and_rejects_missing_ones(self):
        self.assertEqual(app.background_setup("blur")[1], "blur")
        self.assertEqual(app.background_setup(None)[1], "off")
        with self.assertRaisesRegex(RuntimeError, "not found"):
            app.background_setup("/nope/missing.png")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "office.jpg"
            app.cv2.imwrite(str(path), np.zeros((10, 10, 3), np.uint8))
            images, choice = app.background_setup(str(path))
            self.assertEqual(choice, str(path))
            self.assertIn(path, images)

    def test_live_loop_blurs_the_output_but_not_recognition(self):
        frame = self.frame.copy()
        capture = Mock()
        capture.read.return_value = (True, frame)
        camera = Mock(width=160, height=90)
        recognized = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(Mock(), Mock())))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            mocks.enter_context(patch.object(app, "person_mask", return_value=self.mask))
            mocks.enter_context(patch.object(app, "recognize_gesture", side_effect=lambda r, image, ts: (
                recognized.append(image), ("No_Hand", 0.0))[1]))
            mocks.enter_context(patch.object(app, "to_mp_image", side_effect=lambda image: image))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=(0, 0, 0.1, 4.1, 4.3, 4.5)))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows", "imshow"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=(0, 0, 0, ord("q"))))
            mocks.enter_context(patch("builtins.print"))
            with ExitStack() as resources:
                app.run_camera(arguments(background="blur"), resources, "normal")
        sent = camera.send.call_args.args[0]
        np.testing.assert_array_equal(sent[50, 80], frame[50, 80])  # Person stays sharp.
        self.assertFalse(np.array_equal(sent[:, :40], frame[:, :40]))  # Room is blurred.
        self.assertTrue(recognized and all(image is frame for image in recognized))


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

    def test_buttons_wrap_inside_a_narrow_frame_and_goku_keys_return_when_gestures_are_off(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        values = {"b": "meet-ottoman-camp", "c": "MacBook Pro Camera", "g": "off"}
        _, targets = app.draw_hud(frame, "goku", app.ALL_MODES, "", "", "Preview only",
                                  app.HUD_IDLE, True, values)
        for (left, top, right, bottom), _ in targets:
            self.assertTrue(0 <= left < right <= 640 and 0 <= top < bottom <= 360)
        self.assertIn(ord("v"), {key for _, key in targets})
        self.assertGreater(len({box[1] for box, _ in targets}), 2)  # Chips plus 2+ button rows.

    def test_status_when_gestures_are_off(self):
        self.assertIn("press V", app.hud_status("goku", "ready", None, 0.3, None, gestures=False))
        self.assertIn("press R", app.hud_status("itachi", "vanished", None, 0.3, None, gestures=False))

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
