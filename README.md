# VanishCam

Vanish from your webcam with a hand gesture, live in Google Meet, Zoom or any app that
uses a camera. Video is processed locally; nothing is recorded or uploaded. Runs on
macOS, Linux and Windows from source: no installers or executables.

![The six effects, rendered by VanishCam on a drawn scene](docs/demo.gif)

<sub>VanishCam’s effects animate a neon line figure on a black stage (`tools/make_demo_gif.py`).</sub>

## How it works

1. VanishCam captures the empty room while you step out of frame for 4 seconds.
2. You come back and make a gesture. MediaPipe tracks your hands, arms and silhouette
   locally.
3. The effect plays and the captured room shows where you were. In most effects the rest
   of the room stays live.
4. The result goes to a virtual camera, which Meet, Zoom or FaceTime pick like a webcam.

## Effects

Switch with the number keys while it runs.

| Key | Effect | Gesture |
| --- | --- | --- |
| `1` | Normal | V sign (index and middle finger): dissolve |
| `2` | Naruto | Hand seal for 0.3 s: palms pressed together, pointing up, fully in view (chest or face height). The preview says which part is missing |
| `3` | Particles | V sign: you break into drifting particles |
| `4` | Goku | Two joined fingers on the forehead for 0.3 s: teleport; repeat to return |
| `5` | Itachi | Arms crossed over your chest for 0.3 s: flock of crows; cross again to return |
| `6` | Snap | Snap your fingers: puff of smoke; snap again to return in another puff |

Fork of [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam):
English UI, everything switchable live (effect, camera, background), preview-only overlay,
direct virtual-camera output, and no EXE/licensing parts.

## Requirements

- macOS (Apple Silicon or Intel), Linux (x86_64 or arm64) or Windows 10/11 (x86_64)
- Python 3.11–3.13 (MediaPipe has no wheels for newer versions yet)
- A virtual camera driver, so Meet, Zoom and other apps can pick VanishCam as a camera:
  OBS Virtual Camera on macOS and Windows, v4l2loopback on Linux (setup below)

## Install

macOS:

```sh
brew install python@3.12
git clone https://github.com/jakub-tucek/vanishcam.git
cd vanishcam
./vanishcam
```

Linux (Debian/Ubuntu; use your distribution's packages elsewhere):

```sh
sudo apt install python3 python3-venv git libgl1
git clone https://github.com/jakub-tucek/vanishcam.git
cd vanishcam
./vanishcam
```

Windows: install Python 3.12 from [python.org](https://www.python.org/downloads/)
(keep the *py launcher* option on), then in PowerShell or Command Prompt:

```bat
git clone https://github.com/jakub-tucek/vanishcam.git
cd vanishcam
vanishcam
```

The first run creates `.venv`, installs the dependencies from `requirements.txt` (prebuilt
wheels only) and downloads the missing MediaPipe models from Google's
`storage.googleapis.com/mediapipe-models` bucket. Later runs start immediately. The
launchers (`vanishcam` for macOS/Linux, `vanishcam.cmd` for Windows) only pick a Python;
the setup itself is `bootstrap.py`.

Camera access:
- **macOS** asks on the first run; allow your terminal in **System Settings → Privacy &
  Security → Camera**. You can also double-click `VanishCam.command` in Finder.
- **Windows**: **Settings → Privacy & security → Camera → Let desktop apps access your
  camera** must be on. You can also double-click `vanishcam.cmd`.
- **Linux**: your user needs access to `/dev/video*` (usually the `video` group:
  `sudo usermod -aG video $USER`, then log out and in).

## Virtual camera setup (one-time)

VanishCam sends its video to a virtual camera driver, which every app sees as a camera.

### Linux: v4l2loopback

```sh
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="VanishCam Virtual Camera"
```

`exclusive_caps=1` is needed for Chrome (Meet) to list the camera. To load it on every
boot:

```sh
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback devices=1 exclusive_caps=1 card_label="VanishCam Virtual Camera"' \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

In Meet or Zoom pick **VanishCam Virtual Camera**. `C` never switches VanishCam's input to
it (names containing *virtual*, *loopback* or *Dummy video device* are skipped).

### Windows: OBS Virtual Camera

Install [OBS Studio](https://obsproject.com) 30 or later; its installer registers the
**OBS Virtual Camera** driver. Keep the OBS app closed while VanishCam runs (its own
virtual camera would hold the driver). In Meet or Zoom pick **OBS Virtual Camera**.

Windows gives VanishCam no camera names, so `C` cannot recognise the OBS Virtual Camera
by name. If the preview ever shows itself (a picture inside a picture), press `C` again.

### macOS: OBS Virtual Camera

Only the driver is needed; the OBS app stays closed while VanishCam runs. macOS
cannot install that driver without OBS.app, and a self-made driver would need a paid
Apple Developer account.

1. `brew install --cask obs`
2. Open OBS and click **Start Virtual Camera** once.
3. Approve the extension in **System Settings → General → Login Items & Extensions →
   Camera Extensions** (toggle OBS on).
4. Click **Stop Virtual Camera** and **quit OBS** (`Cmd+Q`).
5. If Meet later shows a black picture from *OBS Virtual Camera*, restart the Mac once;
   freshly approved camera extensions sometimes need it.

#### Optional (macOS): shrink OBS to just the driver (~487 MB → ~5 MB)

Once the driver is approved, macOS runs its own copy from `/Library/SystemExtensions/`,
so OBS's bundled frameworks are no longer needed. If you never use OBS itself, you can
hollow out the app and keep only what anchors the driver:

```sh
rm -rf /Applications/OBS.app/Contents/{Frameworks,PlugIns,Resources}
systemextensionsctl list | grep -i obs   # should still say [activated enabled]
```

Keep the rest of `OBS.app` (`MacOS/`, `Info.plist`, `embedded.provisionprofile`,
`_CodeSignature/`, `Library/SystemExtensions/`): macOS removes the driver if the app that
installed it disappears.

- OBS itself no longer starts; don't open it from Launchpad.
- Reinstalling or upgrading OBS restores the full app.
- Tested with OBS 33.0.3 on macOS 26.6: the driver keeps working right after hollowing.
  Behaviour across a reboot is not verified yet.
- To undo, or if the driver stops working: `brew reinstall --cask obs`, open OBS, click
  **Start Virtual Camera** once, quit OBS.

## Use

1. Run `./vanishcam`. The preview opens straight away with your last effect, camera and
   background (Normal on the built-in camera the first time). Change any of them live.
2. Step out of the frame for 4 seconds while the empty background is captured. If you sit
   in a chair, leave the empty chair in place.
3. Step back in and do the gesture shown at the top of the preview.
4. In Meet, Zoom or FaceTime pick the virtual camera (**OBS Virtual Camera**, or your
   v4l2loopback device on Linux). The badge at the top right of the preview says
   **Live: …** when frames are flowing. On Windows run `vanishcam` instead of `./vanishcam`
   in the examples below.

The gestures for each effect are listed under [Effects](#effects).

While vanished (Normal, Particles, Goku, Snap) the room stays live and only your silhouette is
replaced by the captured background, matched to the camera's current exposure.

### Background blur and custom images

Press `B` (or click **B background** in the preview) to cycle: **off → blur → each image**
in `backgrounds/`. The choice is remembered. Drop any `.jpg`, `.png` or `.webp` into
`backgrounds/` (git ignores its contents); images are scaled to fill the frame. VanishCam cuts you out with MediaPipe's general
selfie segmenter, keeps only the regions that are actually you (so lamps, door frames or a
pull-up bar don't show through) and snaps the edge to your hair and face. The style
sits underneath the effects, so smoke, crows and particles still show, and while you are
vanished the whole frame shows the chosen background.

```sh
./vanishcam --background blur
./vanishcam --background ~/Pictures/office.jpg   # or just a file name from backgrounds/
```

To reuse a background you generated in Google Meet: apply it in Meet, step out of the
frame so only the background shows in your self-view, and take a screenshot of it
(`Cmd+Shift+4` on macOS, `Win+Shift+S` on Windows, then crop). Meet does not offer a download, so this is a 720p-ish copy.

### Controls

Keys work while the preview window has focus. The effect chips at the top and the buttons
at the bottom of the preview are clickable too.

| Key | Action |
| --- | --- |
| `1`–`6` | Switch effect (the captured background is kept) |
| `G` | Gestures on/off: when off nothing triggers by itself; use `V` / `R` (Goku too) |
| `V` / `R` | Vanish / reappear manually (in Goku only while gestures are off) |
| `Space` | Recapture the empty room (step out of frame) |
| `C` | Next camera (skips VanishCam's own virtual camera where the OS names cameras) |
| `B` | Background: off → blur → your images |
| `M` | Effect sounds on/off (remembered; `--mute` starts muted) |
| `F` | Fullscreen |
| `H` | Hide or show the button bar |
| `Q` / `Esc` | Quit |

The overlay is drawn on the preview only; Meet gets clean video in the camera's true
orientation (the preview is mirrored, like a mirror).

### Options

```sh
./vanishcam --mode goku --camera 0   # start with this effect and camera index
./vanishcam --goku-delay 1           # Goku hold time in seconds (default 0.3)
./vanishcam --background blur        # off, blur, or an image (default: last choice)
./vanishcam --no-gestures            # start with gestures off (G turns them on)
./vanishcam --no-virtual-camera      # skip the virtual camera
./vanishcam --virtual-camera         # exit if the virtual camera is unavailable
./vanishcam --no-hud                 # plain preview
./vanishcam --no-mirror              # unmirrored preview
./vanishcam --gesture-debug          # recognition details and fps in the terminal
./vanishcam --width 1280 --height 720 --fps 30
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Badge says *Virtual camera busy* | Quit OBS (macOS/Windows) or the other app writing to the loopback device (Linux). VanishCam retries every 3 s. |
| Badge says *No virtual camera driver* | Do the one-time virtual camera setup above. |
| Meet shows black from the virtual camera | Select the camera again or reload the tab; fully quit and reopen Chrome; on macOS restart once after first installing OBS. |
| Chrome doesn't list the camera (Linux) | Reload v4l2loopback with `exclusive_caps=1` (see setup). |
| *Could not access /dev/videoN due to missing permissions* (Linux) | Add yourself to the `video` group and log in again. |
| Meet shows the plain webcam | Meet is set to your real camera; pick the virtual one. |
| *Already running* | Another VanishCam window is open; press `Q` there. |
| *No hand in view* | Hold the gesture higher, around face height, palm towards the camera. |
| You stay visible after vanishing | Press `Space` and stay fully out of frame for the 4-second capture. |
| Windows don't appear | They may open behind the terminal; check `Cmd+Tab` / `Alt+Tab` for Python. |

## Development

```sh
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
./vanishcam --self-test   # checks models, assets and audio without opening a camera
```

Regenerate `docs/demo.gif` after changing an effect:
`.venv/bin/python tools/make_demo_gif.py`.

Layout: `bootstrap.py` (venv setup), `vanish_cam.py` (app, overlay, camera switching), `*_effect.py` (effects),
`vanish_sound.py`, `mediapipe_log_filter.py` (quiets MediaPipe logs), `backgrounds/`, `models/`,
`assets/`, `tests/`, `tools/` (demo GIF).

## Credits

Based on [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam). Uses
Google MediaPipe models (selfie segmenter, gesture recognizer, hand landmarker,
pose landmarker, BlazeFace).
