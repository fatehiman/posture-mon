"""
Posture Monitor
===============

Watches your webcam, learns a "good posture" template (eyes, head/nose and
shoulders), then nags you with a 200 Hz / 0.5 s beep every 5 seconds whenever
any tracked point drifts outside its tolerance.

Flow each run:
    1. DETECTING  - show webcam, find the 5 body points.
    2. CALIBRATE  - sit up straight, click SAVE to store the template.
    3. TOLERANCE  - enter tolerance in cm for eyes / head / shoulders and a
                    blink-reminder time in seconds (0 = off), pre-filled with
                    previously saved values, click SAVE.
    4. MONITOR    - every 5 s, check posture; beep until corrected. If the
                    blink reminder is on, also beep every 3 s whenever you go
                    longer than the set time without blinking.
                    Saved points are GRAY dots, tolerance is a DARK-YELLOW
                    circle, live points are BLUE (inside) or RED (outside).

Distance handling
-----------------
A webcam has no real cm scale and it changes as you move toward/away from the
camera.  We solve both problems by anchoring the saved template to your CURRENT
shoulder midpoint and shoulder width every frame:

    px_per_cm  = current_shoulder_width_px / real_shoulder_width_cm

So leaning in/out just rescales the template (no false alarms), while slouching
moves your eyes/head relative to your shoulder frame -> that is what triggers.
"""

import json
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import mediapipe as mp
except ImportError:
    sys.exit("mediapipe is not installed. Run:  pip install -r requirements.txt")

try:
    import pystray
except ImportError:
    sys.exit("pystray is not installed. Run:  pip install -r requirements.txt")

# Beep is Windows-only via winsound; fall back to a console bell elsewhere.
if sys.platform == "win32":
    import winsound

    def play_beep(freq=200, duration_ms=500):
        # On a background thread so the UI never freezes. Posture uses the
        # default 200 Hz / 500 ms; the blink reminder passes its own tone.
        threading.Thread(
            target=lambda: winsound.Beep(freq, duration_ms), daemon=True
        ).start()
else:
    def play_beep(freq=200, duration_ms=500):
        print("\a", end="", flush=True)


def _config_dir():
    """A stable, writable config location.

    NOTE: in a PyInstaller one-file exe, __file__ lives in a temp unpack folder
    that is deleted on exit, so config must NOT be stored next to __file__ —
    otherwise saved settings vanish and /silent has nothing to resume from.
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "PostureMonitor")
    else:
        d = os.path.join(os.path.expanduser("~"), ".posture_monitor")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.path.dirname(os.path.abspath(sys.argv[0]))
    return d


CONFIG_PATH = os.path.join(_config_dir(), "config.json")
BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else 0

# MediaPipe Pose landmark indices for the points we track.
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
LM = mp_pose.PoseLandmark

# Face Mesh eyelid landmark indices used for the Eye Aspect Ratio (EAR).
# Each list is: [corner, top1, top2, corner, bottom2, bottom1].
EAR_LEFT = [33, 160, 158, 133, 153, 144]
EAR_RIGHT = [362, 385, 387, 263, 373, 380]
# EAR below this means the eye is closed. Open eyes sit ~0.25-0.31 on this
# webcam; a real blink dipped to ~0.06, so 0.20 cleanly separates the two.
BLINK_EAR_THRESHOLD = 0.20
# When a blink is overdue, beep this often until the user blinks.
BLINK_BEEP_INTERVAL_S = 3.0
# Blink reminder tone: higher pitch and shorter than the posture beep so the
# two are easy to tell apart.
BLINK_BEEP_FREQ = 300
BLINK_BEEP_DURATION_MS = 250
# How the blink reminder alerts you. "beep only" plays the tone above; the
# "popup:<corner>" modes instead fade in a borderless eye reminder in that
# screen corner (no beep) until you blink or move the mouse over it.
BLINK_NOTIFY_BEEP = "beep only"
BLINK_NOTIFY_MODES = [
    BLINK_NOTIFY_BEEP,
    "popup:top-right",
    "popup:top-left",
    "popup:bottom-right",
    "popup:bottom-left",
]
# Eye-reminder popup size.
BLINK_POPUP_W, BLINK_POPUP_H = 200, 80
POINTS = {
    "nose": LM.NOSE.value,
    "left_eye": LM.LEFT_EYE.value,
    "right_eye": LM.RIGHT_EYE.value,
    "left_shoulder": LM.LEFT_SHOULDER.value,
    "right_shoulder": LM.RIGHT_SHOULDER.value,
}
# Which tolerance group each tracked point belongs to.
GROUP_OF = {
    "nose": "head",
    "left_eye": "eyes",
    "right_eye": "eyes",
    "left_shoulder": "shoulders",
    "right_shoulder": "shoulders",
}

VISIBILITY_THRESHOLD = 0.6      # landmark confidence to count as "detected"
STABLE_FRAMES_NEEDED = 8        # consecutive good frames before we let you SAVE
MONITOR_INTERVAL_S = 10.0       # how often we check + beep
AUTO_PAUSE_TIMEOUT_S = 10.0     # no body for this long -> auto-pause (user away)
DEFAULT_TOLERANCES = {"eyes": 1.0, "head": 2.0, "shoulders": 5.0}
DEFAULT_SHOULDER_CM = 40.0

# Colors are BGR (OpenCV).
C_GRAY = (160, 160, 160)
C_DARK_YELLOW = (0, 160, 160)
C_BLUE = (255, 80, 0)
C_RED = (0, 0, 255)
C_GREEN = (0, 200, 0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _eye_ear(landmarks, idxs, w, h):
    """Eye Aspect Ratio for one eye: (two vertical lid gaps) / (2 * width)."""
    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs]
    vertical = dist(p[1], p[5]) + dist(p[2], p[4])
    horizontal = dist(p[0], p[3])
    return vertical / (2.0 * horizontal) if horizontal > 1e-6 else 0.0


def list_cameras(max_probe=6):
    """Return a list of (index, name) for available webcams."""
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph
            names = FilterGraph().get_input_devices()
            if names:
                return [(i, n) for i, n in enumerate(names)]
        except Exception:
            pass  # pygrabber missing or failed -> fall back to probing
    cams = []
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, backend)
        if cap is not None and cap.isOpened():
            cams.append((i, f"Camera {i}"))
        if cap is not None:
            cap.release()
    return cams


PREVIEW_W, PREVIEW_H = 360, 270


def center_window(win):
    """Center a window horizontally; place its top at 10% of the screen height
    (the same fixed Y for every window). Its size is unchanged."""
    win.update_idletasks()
    w = win.winfo_reqwidth()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = max(0, (sw - w) // 2)
    y = int(sh * 0.10)
    win.geometry("+%d+%d" % (x, y))


def choose_camera(parent, cams, preselect_index):
    """Modal dialog to pick a webcam, showing a LIVE preview of the selection."""
    top = tk.Toplevel(parent)
    top.title("Select webcam")
    top.grab_set()
    top.resizable(False, False)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
    default = preselect_index if preselect_index is not None else cams[0][0]
    state = {"index": default, "cap": None, "running": True, "after_id": None}

    container = tk.Frame(top, padx=12, pady=12)
    container.pack()
    left = tk.Frame(container)
    left.pack(side="left", anchor="n", padx=(0, 12))
    right = tk.Frame(container)
    right.pack(side="left")

    tk.Label(left, text="Choose a webcam:",
             font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

    var = tk.IntVar(value=default)

    # Preview pane with a black placeholder so layout is stable before first frame.
    placeholder = ImageTk.PhotoImage(
        Image.new("RGB", (PREVIEW_W, PREVIEW_H), (20, 20, 20)))
    preview = tk.Label(right, image=placeholder)
    preview.imgtk = placeholder
    preview.pack()
    pstatus = tk.Label(right, text="", font=("Segoe UI", 9), fg="#c0392b")
    pstatus.pack(pady=(4, 0))

    def open_cam(idx):
        if state["cap"] is not None:
            state["cap"].release()
        cap = cv2.VideoCapture(idx, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        state["cap"] = cap

    def on_select():
        state["index"] = var.get()
        pstatus.config(text="")
        open_cam(state["index"])

    for idx, name in cams:
        tk.Radiobutton(left, text=name, variable=var, value=idx,
                       command=on_select, font=("Segoe UI", 10),
                       anchor="w").pack(fill="x", anchor="w")

    def update_preview():
        if not state["running"]:
            return
        cap = state["cap"]
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
                imgtk = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                preview.imgtk = imgtk
                preview.configure(image=imgtk)
                pstatus.config(text="")
            else:
                pstatus.config(text="No signal from this camera.")
        state["after_id"] = top.after(30, update_preview)

    def confirm():
        state["running"] = False
        if state["after_id"] is not None:
            top.after_cancel(state["after_id"])
        if state["cap"] is not None:
            state["cap"].release()
            state["cap"] = None
        top.destroy()

    tk.Button(left, text="Use this camera", font=("Segoe UI", 11, "bold"),
              command=confirm).pack(pady=(12, 0), anchor="w")
    top.protocol("WM_DELETE_WINDOW", confirm)

    open_cam(state["index"])
    update_preview()
    center_window(top)
    parent.wait_window(top)
    return state["index"]


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


def make_icon_image(size=64, paused=False, grayed=False):
    """Five connected dots (eyes / nose / shoulders) on a rectangular background.

    - ``paused``: draw a red diagonal line across the icon (manually paused).
    - ``grayed``: use a muted gray palette (auto-paused — body not detected).
    """
    from PIL import ImageDraw
    # Gray palette signals "not detected"; blue is the normal active state.
    if grayed:
        bg, fg = (120, 120, 120, 255), (205, 205, 205, 255)
    else:
        bg, fg = (41, 98, 165, 255), (255, 255, 255, 255)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([1, 1, size - 2, size - 2], radius=max(6, size // 7),
                        fill=bg)
    s = size / 64.0

    def P(x, y):
        return (x * s, y * s)

    eye_l, eye_r = P(20, 18), P(44, 18)
    nose = P(32, 34)
    sh_l, sh_r = P(15, 50), P(49, 50)
    eyes_mid = P(32, 18)

    lw = max(2, int(3 * s))
    d.line([eye_l, eye_r], fill=fg, width=lw)     # eyes ----
    d.line([eyes_mid, nose], fill=fg, width=lw)   # | down to nose
    d.line([nose, sh_l], fill=fg, width=lw)       # / to left shoulder
    d.line([nose, sh_r], fill=fg, width=lw)       # \ to right shoulder

    r = max(3, int(5 * s))
    for (x, y) in (eye_l, eye_r, nose, sh_l, sh_r):
        d.ellipse([x - r, y - r, x + r, y + r], fill=fg)

    if paused:
        # Diagonal "disabled" slash: a white halo underneath for contrast,
        # then a bold red line corner-to-corner.
        a, b = P(8, 8), P(56, 56)
        d.line([a, b], fill=(255, 255, 255, 255), width=max(5, int(10 * s)))
        d.line([a, b], fill=(220, 30, 30, 255), width=max(3, int(6 * s)))
    return img


def _load_font(size):
    """Best-effort TrueType font; falls back to PIL's bitmap default."""
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arialbd.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


# Card color for the blink popup; the Toplevel background is set to match so
# the rounded corners blend seamlessly (no gray gaps).
BLINK_POPUP_BG = (20, 40, 70)
BLINK_POPUP_BG_HEX = "#%02x%02x%02x" % BLINK_POPUP_BG


def make_eye_image(w=BLINK_POPUP_W, h=BLINK_POPUP_H):
    """A 'Blink!' reminder card: a drawn eye on the left, text on the right."""
    from PIL import ImageDraw
    img = Image.new("RGBA", (w, h), BLINK_POPUP_BG + (255,))
    d = ImageDraw.Draw(img)
    # Rounded white border just inside the edge.
    d.rounded_rectangle([3, 3, w - 4, h - 4], radius=14,
                        outline=(255, 255, 255, 255), width=2)

    # Eye, centered in the left portion.
    cx, cy = h * 0.62, h * 0.5
    ew, eh = h * 0.38, h * 0.22          # sclera half-width / half-height
    d.ellipse([cx - ew, cy - eh, cx + ew, cy + eh],
              fill=(245, 245, 245, 255), outline=(30, 30, 30, 255), width=1)
    ir = eh * 0.95                        # iris
    d.ellipse([cx - ir, cy - ir, cx + ir, cy + ir], fill=(70, 140, 210, 255))
    pr = ir * 0.5                         # pupil
    d.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=(12, 12, 12, 255))
    hr = pr * 0.4                         # catch-light highlight
    hx, hy = cx - pr * 0.35, cy - pr * 0.35
    d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(255, 255, 255, 255))
    # Upper-lid lash curve over the sclera.
    d.arc([cx - ew, cy - eh, cx + ew, cy + eh], start=200, end=340,
          fill=(30, 30, 30, 255), width=2)

    # "Blink!" text to the right of the eye, vertically centered.
    font = _load_font(int(h * 0.42))
    text = "Blink!"
    try:
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = (h - th) / 2 - bbox[1]
    except Exception:                     # very old Pillow without textbbox
        ty = h * 0.3
    tx = cx + ew + 14
    d.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))
    return img


# Keep the single-instance handle alive for the whole process.
_instance_guard = None


def acquire_single_instance():
    """Return True if we are the only instance; False if another is running."""
    global _instance_guard
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, "PostureMonitor_SingleInstance")
        ERROR_ALREADY_EXISTS = 183
        if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            return False
        _instance_guard = handle
        return True
    # POSIX fallback: bind an abstract-ish localhost port.
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 59123))
    except OSError:
        return False
    _instance_guard = sock
    return True


def pick_camera(parent):
    """List webcams, pre-select the remembered one, prompt if >1, persist choice.
    Returns (index, name) or (None, None) if no camera."""
    cams = list_cameras()
    if not cams:
        messagebox.showerror("Posture Monitor", "No webcam was found.")
        return None, None
    cfg = load_config()
    last_index, last_name = cfg.get("camera_index"), cfg.get("camera_name")
    indices = {i for i, _ in cams}
    preselect = None
    if last_name is not None:
        preselect = next((i for i, n in cams if n == last_name), None)
    if preselect is None and last_index in indices:
        preselect = last_index

    index = choose_camera(parent, cams, preselect) if len(cams) > 1 else cams[0][0]
    name = next((n for i, n in cams if i == index), None)
    cfg["camera_index"], cfg["camera_name"] = index, name
    save_config(cfg)
    return index, name


class PostureMonitor:
    def __init__(self, root, camera_index=0, silent=False):
        self.root = root
        self.camera_index = camera_index
        self.silent = silent
        self.root.title("Posture Monitor")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cfg = load_config()
        self.tolerances = dict(self.cfg.get("tolerances", DEFAULT_TOLERANCES))
        self.real_shoulder_cm = float(
            self.cfg.get("real_shoulder_cm", DEFAULT_SHOULDER_CM)
        )
        # Blink reminder: seconds allowed without a blink (0 = feature off).
        self.blink_reminder_sec = float(self.cfg.get("blink_reminder_sec", 0))
        # How the reminder alerts: "beep only" or "popup:<corner>".
        self.blink_notify_mode = self.cfg.get("blink_notify_mode", BLINK_NOTIFY_BEEP)
        if self.blink_notify_mode not in BLINK_NOTIFY_MODES:
            self.blink_notify_mode = BLINK_NOTIFY_BEEP

        # Saved template: absolute pixel position of each point at SAVE time.
        # These stay FIXED on screen; you move your body back into them.
        # Loaded from config so /silent can resume monitoring with last settings.
        tmpl = self.cfg.get("template")
        self.template = {k: tuple(v) for k, v in tmpl.items()} if tmpl else None
        self.saved_shoulder_px = self.cfg.get("saved_shoulder_px")
        self.state = "DETECTING"
        self.stable_count = 0
        self.last_check_t = 0.0
        self.last_posture_good = True
        self.alive = True
        self.tray = None
        self.paused = False  # when True, suppress all beeps (toggled from tray)
        # Auto-pause: go silent + gray the icon when no body is seen for a
        # while (user away). On by default; reverts the moment a body returns.
        self.auto_pause_enabled = True
        self.auto_paused = False
        self.last_seen_t = None     # last time a body was detected
        self._body_present = False  # set each frame from the pose result

        # Blink tracking. Face Mesh is created lazily, only when the reminder
        # is enabled, so disabling it costs nothing.
        self.face_mesh = None
        self.eye_closed = False          # rising-edge state for blink counting
        self.last_blink_t = None         # when the last blink was seen
        self.last_blink_beep_t = 0.0
        self._last_ear = None
        # Eye-reminder popup (popup:* modes): the live Toplevel + its fade state.
        self.blink_popup = None
        self.blink_popup_alpha = 0.0
        self._eye_photo = None           # cached PhotoImage for the popup
        # One-time flag: center + size the window after the first real frame.
        self._geom_ready = False

        # --- camera ---
        self.cap = cv2.VideoCapture(camera_index, BACKEND)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            if not silent:
                messagebox.showerror(
                    "Posture Monitor", "Could not open the webcam.")
            self.alive = False
            return

        # Silent start: skip calibration UI and monitor with the saved template.
        # If there is no usable saved template, there is nothing to do silently.
        if silent:
            if self.template and self.saved_shoulder_px:
                self.state = "MONITOR"
            else:
                self.cap.release()
                self.alive = False
                return

        self.pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # --- UI ---
        self.status = tk.Label(root, text="", font=("Segoe UI", 13), pady=6)
        self.status.pack(fill="x")

        self.video = tk.Label(root)
        self.video.pack()

        self.controls = tk.Frame(root, pady=8)
        self.controls.pack(fill="x")

        self._build_controls()
        self._show_controls_for_state()

        # Window icon (same five-dot mark as the tray).
        self._icon_photo = ImageTk.PhotoImage(make_icon_image(64))
        self.root.iconphoto(True, self._icon_photo)

        self.update_frame()

    # ------------------------------------------------------------------ UI
    def _build_controls(self):
        # CALIBRATE: a single SAVE button.
        self.btn_save_posture = tk.Button(
            self.controls, text="SAVE good posture",
            font=("Segoe UI", 12, "bold"), width=22, command=self.save_posture
        )

        # TOLERANCE: entry fields + SAVE.
        self.tol_frame = tk.Frame(self.controls)
        self.tol_entries = {}
        for i, grp in enumerate(("eyes", "head", "shoulders")):
            tk.Label(self.tol_frame, text=f"{grp} (cm):",
                     font=("Segoe UI", 11)).grid(row=0, column=i * 2, padx=(8, 2))
            e = tk.Entry(self.tol_frame, width=6, font=("Segoe UI", 11))
            e.insert(0, str(self.tolerances.get(grp, DEFAULT_TOLERANCES[grp])))
            e.grid(row=0, column=i * 2 + 1, padx=(0, 8))
            self.tol_entries[grp] = e

        tk.Label(self.tol_frame, text="shoulder width (cm):",
                 font=("Segoe UI", 11)).grid(row=1, column=0, columnspan=3,
                                             pady=(6, 0), sticky="e")
        self.shoulder_entry = tk.Entry(self.tol_frame, width=6, font=("Segoe UI", 11))
        self.shoulder_entry.insert(0, str(self.real_shoulder_cm))
        self.shoulder_entry.grid(row=1, column=3, pady=(6, 0), sticky="w")

        tk.Label(self.tol_frame, text="blink reminder (sec, 0=off):",
                 font=("Segoe UI", 11)).grid(row=2, column=0, columnspan=3,
                                             pady=(6, 0), sticky="e")
        self.blink_entry = tk.Entry(self.tol_frame, width=6, font=("Segoe UI", 11))
        self.blink_entry.insert(0, str(self.blink_reminder_sec))
        self.blink_entry.grid(row=2, column=3, pady=(6, 0), sticky="w")

        tk.Label(self.tol_frame, text="blink alert:",
                 font=("Segoe UI", 11)).grid(row=3, column=0, columnspan=3,
                                             pady=(6, 0), sticky="e")
        self.blink_mode_combo = ttk.Combobox(
            self.tol_frame, width=18, state="readonly", values=BLINK_NOTIFY_MODES)
        self.blink_mode_combo.set(self.blink_notify_mode)
        self.blink_mode_combo.grid(row=3, column=3, columnspan=3,
                                   pady=(6, 0), sticky="w")

        self.btn_save_tol = tk.Button(
            self.controls, text="SAVE tolerances & start",
            font=("Segoe UI", 12, "bold"), width=22, command=self.save_tolerances
        )

        # MONITOR: recalibrate button.
        self.btn_recal = tk.Button(
            self.controls, text="Recalibrate",
            font=("Segoe UI", 11), width=14, command=self.recalibrate
        )

    def _clear_controls(self):
        for w in (self.btn_save_posture, self.tol_frame,
                  self.btn_save_tol, self.btn_recal):
            w.pack_forget()

    def _show_controls_for_state(self):
        self._clear_controls()
        if self.state == "DETECTING":
            self.status.config(
                text="Detecting your eyes, head and shoulders… stay in view.")
        elif self.state == "CALIBRATE":
            self.status.config(
                text="Sit with GOOD posture, then click SAVE.")
            self.btn_save_posture.pack()
        elif self.state == "TOLERANCE":
            self.status.config(
                text="Set how far each part may move before alerting (cm).")
            # Button first so it lines up with CALIBRATE / MONITOR; fields below.
            self.btn_save_tol.pack()
            self.tol_frame.pack(pady=(8, 0))
        elif self.state == "MONITOR":
            self.btn_recal.pack()

        # Re-fit the minimum size to this state's controls and re-center.
        self.root.after_idle(self._refresh_geometry)

    # -------------------------------------------------------------- actions
    def save_posture(self):
        pts = self._last_points
        if not pts:
            return
        _, width = self._shoulder_frame(pts)
        if width < 1e-3:
            return
        # Store ABSOLUTE pixel positions — these become the fixed on-screen targets.
        self.template = {name: (p[0], p[1]) for name, p in pts.items()}
        self.saved_shoulder_px = width  # used to turn cm tolerances into pixels
        self.state = "TOLERANCE"
        self._show_controls_for_state()

    def save_tolerances(self):
        try:
            tol = {g: float(self.tol_entries[g].get()) for g in self.tol_entries}
            shoulder_cm = float(self.shoulder_entry.get())
            blink_sec = float(self.blink_entry.get())
            if any(v <= 0 for v in tol.values()) or shoulder_cm <= 0:
                raise ValueError
            if blink_sec < 0:  # 0 is allowed (= reminder off), negatives are not
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "Posture Monitor",
                "Tolerances and shoulder width must be positive; "
                "blink reminder must be 0 or more.")
            return
        self.tolerances = tol
        self.real_shoulder_cm = shoulder_cm
        self.blink_reminder_sec = blink_sec
        self.blink_notify_mode = self.blink_mode_combo.get() or BLINK_NOTIFY_BEEP
        self.cfg["tolerances"] = tol
        self.cfg["real_shoulder_cm"] = shoulder_cm
        self.cfg["blink_reminder_sec"] = blink_sec
        self.cfg["blink_notify_mode"] = self.blink_notify_mode
        self._reset_blink_timer()
        # Persist the template so /silent can resume monitoring next time.
        self.cfg["template"] = {
            k: [float(v[0]), float(v[1])] for k, v in self.template.items()
        }
        self.cfg["saved_shoulder_px"] = float(self.saved_shoulder_px)
        save_config(self.cfg)
        self.state = "MONITOR"
        self.last_check_t = 0.0
        self._show_controls_for_state()

    def recalibrate(self):
        self.template = None
        self.saved_shoulder_px = None
        self.stable_count = 0
        self._reset_blink_timer()
        self._reset_auto_pause()
        self.state = "DETECTING"
        self._show_controls_for_state()

    # --------------------------------------------------------------- alerts
    def _beep(self, freq=200, duration_ms=500):
        """Play an alert tone, unless paused (manually or auto)."""
        if self.paused or self.auto_paused:
            return
        play_beep(freq, duration_ms)

    def _refresh_tray_icon(self):
        """Set the tray icon to match the current pause state."""
        if self.tray is None:
            return
        grayed = self.auto_paused and not self.paused
        self.tray.icon = make_icon_image(64, paused=self.paused, grayed=grayed)

    def _update_auto_pause(self, now, body_present):
        """Auto-pause when no body is seen for AUTO_PAUSE_TIMEOUT_S; revert as
        soon as a body reappears. Only active while the toggle is enabled."""
        if not self.auto_pause_enabled:
            if self.auto_paused:                 # toggle was turned off
                self.auto_paused = False
                self._refresh_tray_icon()
            self.last_seen_t = now
            return

        if body_present:
            self.last_seen_t = now
            if self.auto_paused:                 # user came back -> resume
                self.auto_paused = False
                self._refresh_tray_icon()
            return

        if self.last_seen_t is None:
            self.last_seen_t = now
        if not self.auto_paused and now - self.last_seen_t >= AUTO_PAUSE_TIMEOUT_S:
            self.auto_paused = True               # user away -> go silent + gray
            self._refresh_tray_icon()

    # --------------------------------------------------------------- blink
    def _blink_enabled(self):
        return self.blink_reminder_sec > 0

    def _reset_auto_pause(self):
        """Clear any active auto-pause and restore the icon."""
        self.auto_paused = False
        self.last_seen_t = None
        self._refresh_tray_icon()

    def _reset_blink_timer(self):
        """Restart the no-blink countdown from now and silence any reminder."""
        self.eye_closed = False
        self.last_blink_t = None
        self.last_blink_beep_t = 0.0
        self._hide_blink_popup()

    def _blink_mode_is_popup(self):
        return self.blink_notify_mode.startswith("popup:")

    def _ensure_face_mesh(self):
        if self.face_mesh is None:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self.face_mesh

    def _compute_ear(self, rgb, w, h):
        """Average EAR of both eyes for this frame, or None if no face."""
        res = self._ensure_face_mesh().process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return (_eye_ear(lm, EAR_LEFT, w, h) + _eye_ear(lm, EAR_RIGHT, w, h)) / 2.0

    def _run_blink(self, frame, now):
        """Track blinks from the latest EAR and beep when one is overdue."""
        if self.last_blink_t is None:  # first monitored frame: start the clock
            self.last_blink_t = now

        ear = self._last_ear
        if ear is not None:
            if ear < BLINK_EAR_THRESHOLD:
                if not self.eye_closed:        # falling edge = a blink happened
                    self.eye_closed = True
                    self.last_blink_t = now    # restart the countdown
            else:
                self.eye_closed = False

        elapsed = now - self.last_blink_t
        overdue = elapsed >= self.blink_reminder_sec
        if not overdue or self.paused or self.auto_paused:
            # Caught up, or alerts suppressed -> nothing should be on screen.
            self._hide_blink_popup()
        elif self._blink_mode_is_popup():
            self._show_blink_popup()      # idempotent; fades in once
        elif now - self.last_blink_beep_t >= BLINK_BEEP_INTERVAL_S:
            self.last_blink_beep_t = now
            self._beep(BLINK_BEEP_FREQ, BLINK_BEEP_DURATION_MS)

        # Overlay a small blink status on the video.
        if ear is None:
            text, color = "blink: no face", C_GRAY
        elif overdue:
            text, color = "BLINK NOW!", C_RED
        else:
            text, color = f"blink in {self.blink_reminder_sec - elapsed:.0f}s", C_GREEN
        cv2.putText(frame, text, (8, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --------------------------------------------------------- blink popup
    def _blink_popup_xy(self, corner):
        """Top-left screen position for the popup in the requested corner."""
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        m = 28                                  # edge margin
        taskbar = 48                            # keep clear of a bottom taskbar
        x = sw - BLINK_POPUP_W - m if "right" in corner else m
        y = m if "top" in corner else sh - BLINK_POPUP_H - m - taskbar
        return x, y

    def _show_blink_popup(self):
        """Fade in the borderless eye reminder (no-op if one is already up)."""
        if self.blink_popup is not None:
            return
        corner = self.blink_notify_mode.split(":", 1)[1]
        if self._eye_photo is None:
            self._eye_photo = ImageTk.PhotoImage(make_eye_image())

        win = tk.Toplevel(self.root)
        win.overrideredirect(True)              # borderless
        win.configure(bg=BLINK_POPUP_BG_HEX)    # seamless rounded corners
        win.attributes("-topmost", True)
        try:
            win.attributes("-alpha", 0.0)       # start invisible for the fade
        except tk.TclError:
            pass
        x, y = self._blink_popup_xy(corner)
        win.geometry(f"{BLINK_POPUP_W}x{BLINK_POPUP_H}+{x}+{y}")

        lbl = tk.Label(win, image=self._eye_photo, bg=BLINK_POPUP_BG_HEX,
                       borderwidth=0, highlightthickness=0)
        lbl.pack()
        # Mouse moving onto the reminder dismisses it and snoozes the countdown.
        win.bind("<Enter>", lambda _e: self._snooze_blink_popup())
        lbl.bind("<Enter>", lambda _e: self._snooze_blink_popup())

        self.blink_popup = win
        self.blink_popup_alpha = 0.0
        self._fade_in_blink_popup()

    def _fade_in_blink_popup(self):
        if self.blink_popup is None:
            return
        self.blink_popup_alpha = min(1.0, self.blink_popup_alpha + 0.1)
        try:
            self.blink_popup.attributes("-alpha", self.blink_popup_alpha)
        except tk.TclError:
            return
        if self.blink_popup_alpha < 1.0:
            self.root.after(25, self._fade_in_blink_popup)

    def _hide_blink_popup(self):
        if self.blink_popup is not None:
            try:
                self.blink_popup.destroy()
            except tk.TclError:
                pass
            self.blink_popup = None

    def _snooze_blink_popup(self):
        """Mouse touched the popup: dismiss it and restart the no-blink clock."""
        self._hide_blink_popup()
        self.last_blink_t = time.time()

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _shoulder_frame(pts):
        ls, rs = pts["left_shoulder"], pts["right_shoulder"]
        mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        width = dist(ls, rs)
        return mid, width

    def _extract_points(self, landmarks, w, h):
        pts = {}
        for name, idx in POINTS.items():
            lm = landmarks[idx]
            if lm.visibility < VISIBILITY_THRESHOLD:
                return None
            pts[name] = (lm.x * w, lm.y * h)
        return pts

    # ---------------------------------------------------------- main loop
    def update_frame(self):
        if not self.alive:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)  # mirror, feels natural
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        pts = None
        if result.pose_landmarks:
            pts = self._extract_points(result.pose_landmarks.landmark, w, h)
        self._last_points = pts
        # A body is "present" if MediaPipe found any pose at all (used by
        # auto-pause to decide whether the user is at the computer).
        self._body_present = result.pose_landmarks is not None

        # Blink detection only runs while monitoring and only if enabled.
        if self.state == "MONITOR" and self._blink_enabled():
            self._last_ear = self._compute_ear(rgb, w, h)
        else:
            self._last_ear = None

        if self.state == "DETECTING":
            self._run_detecting(frame, pts)
        elif self.state == "CALIBRATE":
            self._draw_points(frame, pts, C_GREEN)
            if pts is None:  # lost detection -> go back
                self.stable_count = 0
                self.state = "DETECTING"
                self._show_controls_for_state()
        elif self.state == "TOLERANCE":
            # Live preview: fixed saved dots + circles sized by the values being typed.
            tol, shoulder_cm = self._read_live_tolerances()
            self._draw_posture(frame, pts, tol, shoulder_cm)
        elif self.state == "MONITOR":
            self._run_monitor(frame, pts)

        self._render(frame)
        if not self._geom_ready:
            # First frame gives the video its true size -> now we can size+center.
            self._geom_ready = True
            self.root.after_idle(self._refresh_geometry)
        self.root.after(15, self.update_frame)

    def _run_detecting(self, frame, pts):
        self._draw_points(frame, pts, C_GREEN)
        if pts is not None:
            self.stable_count += 1
            self.status.config(
                text=f"Detecting… ({self.stable_count}/{STABLE_FRAMES_NEEDED})")
            if self.stable_count >= STABLE_FRAMES_NEEDED:
                self.state = "CALIBRATE"
                self._show_controls_for_state()
        else:
            self.stable_count = 0
            self.status.config(
                text="Can't see you — make sure head & shoulders are in frame.")

    def _read_live_tolerances(self):
        """Read tolerance/shoulder entry fields, falling back to saved values."""
        tol = {}
        for grp, entry in self.tol_entries.items():
            try:
                tol[grp] = max(0.0, float(entry.get()))
            except ValueError:
                tol[grp] = self.tolerances.get(grp, DEFAULT_TOLERANCES[grp])
        try:
            shoulder_cm = float(self.shoulder_entry.get())
            if shoulder_cm <= 0:
                raise ValueError
        except ValueError:
            shoulder_cm = self.real_shoulder_cm
        return tol, shoulder_cm

    def _draw_posture(self, frame, pts, tolerances, real_shoulder_cm):
        """Draw FIXED saved dots + tolerance circles, and the live moving points.
        Returns True if every visible point is inside its circle."""
        if self.template is None or self.saved_shoulder_px is None:
            return True
        px_per_cm = self.saved_shoulder_px / real_shoulder_cm
        all_inside = True
        for name, saved in self.template.items():
            sx, sy = int(saved[0]), int(saved[1])
            tol_px = tolerances[GROUP_OF[name]] * px_per_cm
            cv2.circle(frame, (sx, sy), max(1, int(tol_px)), C_DARK_YELLOW, 2)  # tolerance
            cv2.circle(frame, (sx, sy), 5, C_GRAY, -1)                          # saved dot
            if pts is not None:
                cur = pts[name]
                inside = dist(cur, saved) <= tol_px
                cv2.circle(frame, (int(cur[0]), int(cur[1])), 7,
                           C_BLUE if inside else C_RED, -1)                     # live point
                if not inside:
                    all_inside = False
            else:
                all_inside = False
        return all_inside

    def _run_monitor(self, frame, pts):
        all_inside = self._draw_posture(
            frame, pts, self.tolerances, self.real_shoulder_cm)

        now = time.time()

        # Auto-pause check: silence + gray the icon when the user is away.
        self._update_auto_pause(now, self._body_present)

        # Blink reminder runs independently of posture — it works off the face,
        # so keep it going even when the shoulders drop out of frame.
        if self._blink_enabled():
            self._run_blink(frame, now)

        if pts is None:
            self.status.config(text="Monitoring — can't see you right now.",
                               fg="#c0392b")
            return

        self.last_posture_good = all_inside

        # Check + beep on the 5-second cadence.
        if now - self.last_check_t >= MONITOR_INTERVAL_S:
            self.last_check_t = now
            if not all_inside:
                self._beep()

        if all_inside:
            self.status.config(text="✓ Good posture", fg="#1a7f1a")
        else:
            self.status.config(text="✗ Fix your posture!", fg="#c0392b")

    # ------------------------------------------------------------- drawing
    @staticmethod
    def _draw_points(frame, pts, color):
        if not pts:
            return
        for p in pts.values():
            cv2.circle(frame, (int(p[0]), int(p[1])), 6, color, -1)

    def _render(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video.imgtk = imgtk            # keep a reference
        self.video.configure(image=imgtk)

    # --------------------------------------------------------- tray / window
    def start_tray(self, notify=None):
        menu = pystray.Menu(
            pystray.MenuItem("Show/Hide", self._tray_toggle, default=True),
            pystray.MenuItem("Paused", self._tray_toggle_pause,
                             checked=lambda item: self.paused),
            pystray.MenuItem("Auto pause", self._tray_toggle_auto_pause,
                             checked=lambda item: self.auto_pause_enabled),
            pystray.MenuItem("Reset", self._tray_reset),
            pystray.MenuItem("Exit", self._tray_exit),
        )
        self.tray = pystray.Icon(
            "PostureMonitor", make_icon_image(64), "Posture Monitor", menu)

        def runner():
            def setup(icon):
                icon.visible = True
                if notify:
                    try:
                        icon.notify(notify, "Posture Monitor")
                    except Exception:
                        pass
            self.tray.run(setup=setup)

        threading.Thread(target=runner, daemon=True).start()

    # Tray callbacks run on pystray's thread -> marshal onto the Tk main loop.
    def _tray_toggle(self, *_):
        self.root.after(0, self.toggle_window)

    def _tray_toggle_pause(self, icon, item):
        # Pausing only flips a flag + swaps the tray icon — no Tk involved, so
        # it's safe to run on pystray's own thread.
        self.paused = not self.paused
        self._refresh_tray_icon()
        icon.update_menu()  # refresh the check mark

    def _tray_toggle_auto_pause(self, icon, item):
        self.auto_pause_enabled = not self.auto_pause_enabled
        if not self.auto_pause_enabled:   # disabling clears any active auto-pause
            self.auto_paused = False
            self.last_seen_t = None
        self._refresh_tray_icon()
        icon.update_menu()

    def _tray_reset(self, *_):
        self.root.after(0, self.do_reset)

    def _tray_exit(self, *_):
        self.root.after(0, self.do_exit)

    def toggle_window(self):
        # Clicking the tray icon shows the window if hidden, hides it if shown.
        try:
            visible = bool(self.root.winfo_viewable())
        except tk.TclError:
            visible = False
        if visible:
            self.root.withdraw()
        else:
            self.show_window()

    def _refresh_geometry(self):
        """Set the window's minimum size to fit the current controls (so the
        tolerance fields can never be clipped) and re-position it: centered
        horizontally with its top fixed at 10% of the screen height, so the
        button stays put across all three states."""
        if not self.alive:
            return
        try:
            self.root.update_idletasks()
            w = self.root.winfo_reqwidth()
            h = self.root.winfo_reqheight()
        except tk.TclError:
            return
        if w <= 1 or h <= 1:
            return
        # Minimum = required size for this state: shrinking can't hide fields.
        self.root.minsize(w, h)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = int(sh * 0.10)
        try:
            self.root.geometry("+%d+%d" % (x, y))
        except tk.TclError:
            pass

    def show_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def on_close(self):
        # X button hides to tray rather than quitting; Exit (tray) really quits.
        self.root.withdraw()

    def do_reset(self):
        """Start from scratch: re-ask the webcam, recalibrate. Keep saved values."""
        self.cap.release()
        index, name = pick_camera(self.root)
        if index is None:                      # no camera -> reopen the old one
            self.cap = cv2.VideoCapture(self.camera_index, BACKEND)
        else:
            self.camera_index = index
            self.cfg["camera_index"], self.cfg["camera_name"] = index, name
            self.cap = cv2.VideoCapture(index, BACKEND)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Recalibrate posture, but keep remembered tolerances / shoulder width.
        self.template = None
        self.saved_shoulder_px = None
        self.stable_count = 0
        self._reset_blink_timer()
        self._reset_auto_pause()
        self.state = "DETECTING"
        self._show_controls_for_state()
        self.show_window()

    def do_exit(self):
        self.alive = False
        self._hide_blink_popup()
        if self.tray is not None:
            try:
                self.tray.stop()
            except Exception:
                pass
        try:
            self.cap.release()
            self.pose.close()
            if self.face_mesh is not None:
                self.face_mesh.close()
        finally:
            self.root.destroy()


def _dbg(msg):
    if os.environ.get("POSTURE_DEBUG"):
        sys.stderr.write(f"[posture] {msg}\n")
        sys.stderr.flush()


def main():
    silent = any(a.lower().lstrip("-/") == "silent" for a in sys.argv[1:])
    _dbg(f"argv={sys.argv[1:]} silent={silent}")

    if not acquire_single_instance():
        _dbg("another instance is running -> exit")
        if not silent:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Posture Monitor", "Posture Monitor is already running.")
            root.destroy()
        return

    root = tk.Tk()
    root.withdraw()

    if silent:
        # No popups, no window: resume monitoring with the last saved settings.
        cfg = load_config()
        has_profile = (cfg.get("camera_index") is not None and cfg.get("template")
                       and cfg.get("saved_shoulder_px") is not None)
        _dbg(f"silent: has_profile={has_profile}")
        if has_profile:
            app = PostureMonitor(
                root, camera_index=cfg["camera_index"], silent=True)
            if not app.alive:
                root.destroy()
                return
            app.start_tray(notify="Posture monitoring started.")
            root.mainloop()
            return
        # No profile yet: ignore /silent and fall through to interactive setup
        # (camera selection -> calibrate -> tolerances).

    # Interactive: choose camera, then run the detect -> calibrate -> monitor flow.
    _dbg("entering interactive pick_camera")
    index, _ = pick_camera(root)
    _dbg(f"pick_camera returned index={index}")
    if index is None:
        root.destroy()
        return

    app = PostureMonitor(root, camera_index=index)
    _dbg(f"app constructed alive={app.alive}")
    if not app.alive:
        root.destroy()
        return
    app.start_tray()
    root.deiconify()
    _dbg("interactive mainloop")
    root.mainloop()


if __name__ == "__main__":
    main()
