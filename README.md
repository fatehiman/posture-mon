# Posture Monitor

Watches your webcam, learns your **good posture** (eyes, head/nose, shoulders),
and beeps at you when you slouch. It can also remind you to **blink** — useful
for preventing dry eyes during long screen sessions. Runs as a **system-tray
app**.

## System tray

The app lives in the system tray (its icon is five connected dots — eyes, nose
and shoulders — on a blue rectangle). **Right-click** the icon for:

- **Show** — bring the window back (also the default double-click action).
- **Reset** — start over: re-ask which webcam to use and re-calibrate your good
  posture. Your remembered values (tolerances, shoulder width, last camera) are
  **kept**, not cleared.
- **Exit** — quit the app.

Closing the window with the **X** button just **hides it to the tray** —
monitoring keeps running. Only **Exit** quits.

**Single instance:** launching a second copy shows "Posture Monitor is already
running." and exits.

## Silent mode

Run with the **`/silent`** argument (also `--silent`) to start with **no popups
and no window** — it goes straight to the tray and begins monitoring using your
**last saved settings** (camera, calibrated posture, tolerances):

```
PostureMonitor.exe /silent
```

Ideal for auto-start on login. If there is **no saved profile yet**, the
`/silent` argument is **ignored** and the app opens normally, starting with the
webcam-selection window so you can calibrate. After you've saved a profile once,
`/silent` runs straight to the tray. To start with Windows, drop a shortcut to
`PostureMonitor.exe /silent` in your Startup folder (`shell:startup`).

## How it works

Every run goes through four stages, all in one window:

1. **Detecting** — the app finds your 5 body points (left/right eye, nose, left/right shoulder). Keep your head and shoulders in frame.
2. **Calibrate** — sit up straight and click **SAVE good posture**. Your template is stored.
3. **Tolerance** — enter how far each part may drift before alerting, in cm (pre-filled with your last saved values). Defaults: `eyes = 1`, `head = 2`, `shoulders = 5`. Also set the **blink reminder (sec)** here — see below. Click **SAVE tolerances & start**.
4. **Monitor** — every **10 seconds** the app checks your posture. If any point is outside its tolerance it plays a **200 Hz, 0.5 s beep**, and keeps beeping each interval until you fix it.

## Blink reminder (dry-eye prevention)

On the **Tolerance** screen there's a **blink reminder (sec)** field:

- **`0` = off** (no blink monitoring).
- **A number, e.g. `10`** = if you go **10 seconds without blinking**, the app
  plays a **300 Hz / 0.25 s beep every 3 seconds** until you blink (a higher,
  shorter tone than the 200 Hz / 0.5 s posture beep, so they're easy to tell
  apart). The moment you blink, the timer resets and starts counting toward the
  next 10 s.

Blinks are detected with MediaPipe **Face Mesh** using the **Eye Aspect Ratio**
(the eye "closes" when the ratio drops below a threshold). While monitoring, a
small status is overlaid at the bottom of the video: a countdown
(`blink in Ns`), **`BLINK NOW!`** in red when overdue, or `blink: no face` if
your eyes aren't visible. Face Mesh only runs when the reminder is enabled, so
leaving it at `0` costs no extra CPU.

In monitoring view:

- **Gray dots** = your saved good-posture points.
- **Dark-yellow circles** = the allowed tolerance radius around each point.
- **Blue dot** = a live point that is *inside* its circle. ✓
- **Red dot** = a live point that is *outside* its circle. ✗ → move that body part back into its circle.

## Fixed targets

The saved gray dots and dark-yellow circles are **fixed in place** at the exact
pixel positions captured when you clicked SAVE. They do **not** follow your body.
The live blue/red points move with you, and the goal is to move your body so each
live point sits back inside its fixed circle.

A webcam only knows pixels, so cm tolerances are converted using the shoulder
width measured at SAVE time:

```
px_per_cm = saved_shoulder_width_px / real_shoulder_width_cm
```

`real_shoulder_width_cm` defaults to **40 cm** and is editable on the tolerance
screen — as you type, the circles update **live** in the preview. Measure
shoulder-to-shoulder once for best accuracy. Tolerances and shoulder width
persist in `config.json`.

> **Note:** because the targets are fixed, if you move your chair to a noticeably
> different position/distance, click **Recalibrate** to capture a fresh template.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python posture_monitor.py
```

**Python 3.10–3.12** is recommended (MediaPipe wheels). On non-Windows the beep
falls back to a terminal bell.

## Building the .exe

A standalone Windows executable (no Python needed) is built with PyInstaller:

```powershell
py -3.10 -m pip install pyinstaller
.\build_exe.ps1
```

The result is `dist\PostureMonitor.exe` — a single windowed file with the
five-dot icon. (First launch is a little slow as the one-file bundle unpacks.)

## Files

- `posture_monitor.py` — the whole app.
- `requirements.txt` — dependencies.
- `build_exe.ps1` — PyInstaller build script.
- `icon.ico` — app icon (generated from the five-dot mark).
- `config.json` — remembered values (camera, tolerances, shoulder width, calibrated posture, blink-reminder seconds). Stored in **`%APPDATA%\PostureMonitor\config.json`** so it persists for both the script and the one-file exe.
- `dist\PostureMonitor.exe` — the built executable.
