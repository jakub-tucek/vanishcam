# VanishCam for macOS

Vanish from your webcam with a hand gesture, live in Google Meet, Zoom or any app that
uses a camera. Video is processed locally; nothing is recorded or uploaded.

macOS fork of [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam):
English UI, one-window launcher, live effect switching, preview-only status overlay,
direct virtual-camera output, and no Windows/EXE/licensing parts.

## Requirements

- macOS on Apple Silicon or Intel
- [Homebrew](https://brew.sh)
- Python 3.11–3.13 (MediaPipe has no wheels for newer versions yet)
- Google Meet: Chrome, Edge, Brave or Arc with the bundled extension (no driver needed)
- Desktop apps (Zoom app, FaceTime, ...): the OBS Virtual Camera driver, optional

## Install

```sh
brew install python@3.12
git clone https://github.com/jakub-tucek/vanishcam-mac.git
cd vanishcam-mac
./vanishcam
```

The first `./vanishcam` run creates `.venv`, installs the dependencies from
`requirements.txt` (prebuilt wheels only) and downloads the missing MediaPipe models from
Google's `storage.googleapis.com/mediapipe-models` bucket. Later runs start immediately.

macOS asks for **camera access** for your terminal on the first run; allow it in
**System Settings → Privacy & Security → Camera**.

You can also double-click `VanishCam.command` in Finder.

## Google Meet in Chrome, no OBS (one-time)

The `browser-extension/` folder adds a **VanishCam** camera to Google Meet. With
`--browser`, VanishCam serves its video on `127.0.0.1` to the extension on
`https://meet.google.com`; other websites cannot read it.

The server is **off by default**: while it runs, any program on your Mac (and other users
logged into it) could read the processed video without macOS camera permission. Enable it
only when you need Meet in the browser.

1. Open `chrome://extensions`, turn on **Developer mode** (top right).
2. Click **Load unpacked** and pick the `browser-extension` folder of this repo.
3. Start `./vanishcam --browser`, open Meet, and choose
   **Settings → Video → Camera → VanishCam**.
4. Chrome asks once whether meet.google.com may access apps and services on this device:
   click **Allow**. (That is how the page reaches VanishCam on `127.0.0.1`.)

Your microphone keeps working as usual. Without the app running, the VanishCam camera shows
"Start the VanishCam app".

## Desktop apps via the OBS driver (optional, one-time)

For the Zoom desktop app, FaceTime and other native apps, VanishCam also writes to the
**OBS Virtual Camera** driver when it is installed. Only the driver is needed; the OBS app
stays closed while VanishCam runs. macOS cannot install that driver without OBS.app, and
a self-made driver would need a paid Apple Developer account.

1. `brew install --cask obs`
2. Open OBS and click **Start Virtual Camera** once.
3. Approve the extension in **System Settings → General → Login Items & Extensions →
   Camera Extensions** (toggle OBS on).
4. Click **Stop Virtual Camera** and **quit OBS** (`Cmd+Q`).
5. If Meet later shows a black picture from *OBS Virtual Camera*, restart the Mac once;
   freshly approved camera extensions sometimes need it.

### Optional: shrink OBS to just the driver (~487 MB → ~5 MB)

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

1. Run `./vanishcam`. In the **VanishCam** window pick an effect (`1`–`5` or click) and a
   webcam (`A`/`D` or `-`/`+`), then press **Enter**. The choice is remembered.
2. Step out of the frame for 4 seconds while the empty background is captured. If you sit
   in a chair, leave the empty chair in place.
3. Step back in and do the gesture shown at the top of the preview.
4. In Meet pick the camera **VanishCam** (or **OBS Virtual Camera** in desktop apps). The
   badge at the top right of the preview says **Live: browser stream** or
   **Live: OBS Virtual Camera** when frames are flowing.

| Key | Effect | Gesture |
| --- | --- | --- |
| `1` | Normal | V sign (index and middle finger): dissolve |
| `2` | Naruto | Palms pressed together: smoke and a falling log |
| `3` | Particles | V sign: you break into drifting particles |
| `4` | Goku | Two joined fingers on the forehead for 0.3 s: teleport; repeat to return |
| `5` | Itachi | Both hands touching at the chest: flock of crows; repeat to return |

While vanished (Normal, Particles, Goku) the room stays live and only your silhouette is
replaced by the captured background, matched to the camera's current exposure.

### Controls

Keys work while the preview window has focus. The effect chips at the top and the buttons
at the bottom of the preview are clickable too.

| Key | Action |
| --- | --- |
| `1`–`5` | Switch effect (the captured background is kept) |
| `V` / `R` | Vanish / reappear manually (not in Goku) |
| `Space` | Capture a new background |
| `F` | Fullscreen |
| `H` | Hide or show the button bar |
| `Q` / `Esc` | Quit |

The overlay is drawn on the preview only; Meet gets clean video in the camera's true
orientation (the preview is mirrored, like a mirror).

### Options

```sh
./vanishcam --mode goku --camera 0   # skip the launcher
./vanishcam --goku-delay 1           # Goku hold time in seconds (default 0.3)
./vanishcam --no-virtual-camera      # skip the OBS driver
./vanishcam --browser                # serve the Meet browser extension (off by default)
./vanishcam --virtual-camera         # exit if the virtual camera is unavailable
./vanishcam --no-hud                 # plain preview
./vanishcam --no-mirror              # unmirrored preview
./vanishcam --gesture-debug          # recognition details and fps in the terminal
./vanishcam --width 1280 --height 720 --fps 30
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| No *VanishCam* camera in Meet | Check the extension is enabled in `chrome://extensions`, then reload the Meet tab. |
| Meet shows "Start the VanishCam app" | Start `./vanishcam --browser`. If it is running, allow meet.google.com to access apps on this device (padlock icon → Site settings → Local network access). |
| Meet shows the plain webcam | Meet is set to *MacBook Pro Camera*; pick *VanishCam*. |
| Desktop app shows black from *OBS Virtual Camera* | Quit OBS (its own virtual camera holds the driver); restart the Mac once after first installing OBS. |
| *Already running* | Another VanishCam window is open; press `Q` there. |
| *No hand in view* | Hold the gesture higher, around face height, palm towards the camera. |
| You stay visible after vanishing | Press `Space` and stay fully out of frame for the 4-second capture. |
| Windows don't appear | They may open behind the terminal; check `Cmd+Tab` for Python. |

## Development

```sh
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
./vanishcam --self-test   # checks models, assets and audio without opening a camera
```

Layout: `vanish_cam.py` (app, launcher, overlay), `*_effect.py` (effects),
`vanish_sound.py`, `mediapipe_log_filter.py` (quiets MediaPipe logs), `browser_output.py`
(localhost frame server) with `browser-extension/` (Meet camera), `models/`, `assets/`,
`tests/`.

## Credits

Based on [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam). Uses
Google MediaPipe models (selfie segmenter, gesture recognizer, hand landmarker,
BlazeFace).
