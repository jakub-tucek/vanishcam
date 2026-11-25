import pytest

import vanish_cam


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Tests must never read or overwrite the user's real ~/.config/vanishcam settings."""
    monkeypatch.setattr(vanish_cam, "SETTINGS_PATH", tmp_path / "settings.json")


@pytest.fixture(autouse=True)
def no_real_camera_list(monkeypatch):
    """Keep tests independent of the cameras attached to this machine."""
    monkeypatch.setattr(vanish_cam, "camera_names", lambda: [])


@pytest.fixture(autouse=True)
def synchronous_gestures(monkeypatch):
    """Loop tests use scripted frame times, so recognition must finish within its frame."""
    monkeypatch.setattr(vanish_cam, "GESTURE_THREAD", False)
