"""Create .venv on first run, keep it in sync with requirements.txt, then start VanishCam.

Runs on macOS, Linux and Windows; the vanishcam / vanishcam.cmd scripts only pick a Python.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
# Written only after a successful install, so a failed first run is retried.
STAMP = VENV / ".requirements.sha256"
SUPPORTED = ((3, 11), (3, 13))


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> int:
    if not SUPPORTED[0] <= sys.version_info[:2] <= SUPPORTED[1]:
        print(f"[VanishCam] Python 3.11-3.13 is required (MediaPipe); this is "
              f"{sys.version.split()[0]}.", file=sys.stderr)
        return 1
    python = venv_python()
    if not python.exists():
        print(f"[VanishCam] First run: creating .venv with Python {sys.version.split()[0]}...")
        venv.create(VENV, with_pip=True)
    wanted = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not STAMP.is_file() or STAMP.read_text().strip() != wanted:
        print("[VanishCam] Installing dependencies...")
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                        "--only-binary=:all:", "-r", str(REQUIREMENTS)], check=True)
        STAMP.write_text(wanted + "\n")
    command = [str(python), str(ROOT / "vanish_cam.py"), *sys.argv[1:]]
    if os.name == "nt":
        try:
            return subprocess.call(command)
        except KeyboardInterrupt:
            return 130
    os.execv(command[0], command)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        sys.exit(f"[VanishCam] Installing dependencies failed (exit {error.returncode}).")
