import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import vanish_cam as app


class LinuxCameraListTests(unittest.TestCase):
    def test_names_follow_dev_video_numbers_and_fill_gaps(self):
        with TemporaryDirectory() as root:
            for number, name in ((0, "Integrated Camera"), (1, "Integrated Camera"),
                                 (3, "VanishCam Virtual Camera")):
                node = Path(root) / f"video{number}"
                node.mkdir()
                (node / "name").write_text(name + "\n")
            (Path(root) / "videoX").mkdir()
            self.assertEqual(app.linux_camera_names(Path(root)),
                             ["Integrated Camera", "Integrated Camera", "video2",
                              "VanishCam Virtual Camera"])

    def test_no_devices(self):
        with TemporaryDirectory() as root:
            self.assertEqual(app.linux_camera_names(Path(root)), [])


class VirtualCameraNameTests(unittest.TestCase):
    def test_own_outputs_are_never_inputs(self):
        for name in ("OBS Virtual Camera", "VanishCam Virtual Camera", "Dummy video device (0x0000)",
                     "v4l2loopback device"):
            with self.subTest(name):
                self.assertTrue(app.is_virtual_camera(name))
        for name in ("MacBook Pro Camera", "Integrated Camera", "HD Pro Webcam C920"):
            with self.subTest(name):
                self.assertFalse(app.is_virtual_camera(name))


class InstanceLockTests(unittest.TestCase):
    def test_second_lock_in_another_process_fails(self):
        import os
        import subprocess
        import sys
        # A private temp dir, so a running VanishCam's lock is neither hit nor blocked.
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        env = {**os.environ, "TMPDIR": temp.name, "TMP": temp.name, "TEMP": temp.name}
        code = "import vanish_cam as app; print(app.acquire_instance_lock())"
        root = str(Path(app.__file__).parent)
        holder = subprocess.Popen(
            [sys.executable, "-c", code + "; import sys; sys.stdout.flush(); sys.stdin.read()"],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(holder.stdout.readline().strip(), "True")
            second = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(second.stdout.strip(), "False")
        finally:
            holder.communicate("", timeout=60)


if __name__ == "__main__":
    unittest.main()
