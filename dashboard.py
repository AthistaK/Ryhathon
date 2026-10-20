import tkinter as tk
from tkinter import ttk
import threading
import queue
import time

import cv2
import pyttsx3
from PIL import Image, ImageTk
from ultralytics import YOLO


# ASTRIX classroom-demo dashboard
# Keep detect.py and test_hazards.py unchanged.

MODEL_PATH = "yolo11n.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.25
ZONE_LEFT = 0.35
ZONE_RIGHT = 0.65
WARNING_REPEAT_SECONDS = 2.5

BG = "#0b1220"
PANEL = "#121d30"
PANEL_LIGHT = "#1a2940"
TEXT = "#f3f6fc"
MUTED = "#a8b5c9"
BLUE = "#6ea8fe"
GREEN = "#42d392"
RED = "#ff6b6b"
AMBER = "#ffc857"

root = tk.Tk()
root.title("ASTRIX | Offline Obstacle Warning System")
root.geometry("1180x760")
root.minsize(900, 620)
root.configure(bg=BG)

model = None
camera = None
running = False
after_id = None
photo_ref = None

speech_queue = queue.Queue(maxsize=1)
speech_stop = threading.Event()
voice_thread = None

last_warning_time = 0.0
warning_active = False
latest_objects = []
frame_count = 0
fps_value = 0.0
fps_start = time.monotonic()


def speak_warning_worker():
    """Use the same reinitialise-per-warning strategy as the working prototype."""
    while not speech_stop.is_set():
        try:
            message = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if message is None:
            speech_queue.task_done()
            break

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 1.0)
            engine.say(message)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as error:
            print("Voice error:", error)
        finally:
            speech_queue.task_done()


def queue_warning(message):
    try:
        if speech_queue.empty():
            speech_queue.put_nowait(message)
    except queue.Full:
        pass


def make_label(parent, text, size=11, color=TEXT, bold=False, **kwargs):
    return tk.Label(
        parent,
        text=text,
        font=("Segoe UI", size, "bold" if bold else "normal"),
        bg=kwargs.pop("bg", PANEL),
        fg=color,
        **kwargs
    )


header = tk.Frame(root, bg=BG, padx=22, pady=16)
header.pack(fill="x")

brand = tk.Frame(header, bg=BG)
brand.pack(side="left", fill="x", expand=True)
make_label(brand, "ASTRIX", 24, TEXT, True, bg=BG).pack(anchor="w")
make_label(
    brand,
    "OFFLINE OBSTACLE WARNING SYSTEM",
    9,
    MUTED,
    True,
    bg=BG
).pack(anchor="w", pady=(2, 0))

connection_badge = make_label(
    header, "●  LOCAL MODE", 10, GREEN, True,
    bg=PANEL, padx=12, pady=8
)
connection_badge.pack(side="right", padx=(10, 0))

main = tk.Frame(root, bg=BG, padx=22)
main.pack(fill="both", expand=True)

left = tk.Frame(main, bg=BG)
left.pack(side="left", fill="both", expand=True, padx=(0, 14))

right = tk.Frame(main, bg=BG, width=285)
right.pack(side="right", fill="y")
right.pack_propagate(False)

camera_card = tk.Frame(left, bg=PANEL, padx=12, pady=12)
camera_card.pack(fill="both", expand=True)

camera_heading = tk.Frame(camera_card, bg=PANEL)
camera_heading.pack(fill="x", pady=(0, 10))
make_label(camera_heading, "LIVE CAMERA", 12, TEXT, True).pack(side="left")
camera_state = make_label(camera_heading, "INITIALISING", 9, AMBER, True)
camera_state.pack(side="right")

video_panel = tk.Frame(camera_card, bg="#050912")
video_panel.pack(fill="both", expand=True)
video_label = tk.Label(
    video_panel,
    text="Loading AI model and webcam...",
    font=("Segoe UI", 13),
    bg="#050912",
    fg=MUTED
)
video_label.pack(fill="both", expand=True)

footer_camera = make_label(
    camera_card,
    "Press Q or click STOP to end the demo",
    9,
    MUTED
)
footer_camera.pack(anchor="w", pady=(9, 0))

status_card = tk.Frame(right, bg=PANEL, padx=15, pady=15)
status_card.pack(fill="x", pady=(0, 12))
make_label(status_card, "MONITORING STATUS", 10, MUTED, True).pack(anchor="w")
warning_status = make_label(
    status_card, "STARTING...", 18, AMBER, True, bg=PANEL,
    wraplength=245, justify="left"
)
warning_status.pack(anchor="w", pady=(10, 5))
status_detail = make_label(
    status_card, "Initialising local detection", 9, MUTED,
    bg=PANEL, wraplength=245, justify="left"
)
status_detail.pack(anchor="w")

objects_card = tk.Frame(right, bg=PANEL, padx=15, pady=15)
objects_card.pack(fill="both", expand=True, pady=(0, 12))
make_label(objects_card, "DETECTED OBJECTS", 10, MUTED, True).pack(anchor="w")
object_count = make_label(objects_card, "0 objects", 16, TEXT, True)
object_count.pack(anchor="w", pady=(8, 10))

objects_list = tk.Listbox(
    objects_card,
    bg=PANEL_LIGHT,
    fg=TEXT,
    selectbackground=BLUE,
    selectforeground="#08111f",
    font=("Consolas", 10),
    relief="flat",
    borderwidth=0,
    highlightthickness=0,
    height=9
)
objects_list.pack(fill="both", expand=True)

make_label(
    objects_card,
    "Confidence is the model score, not a measure of distance or safety.",
    8,
    MUTED,
    bg=PANEL,
    wraplength=245,
    justify="left"
).pack(anchor="w", pady=(10, 0))

voice_card = tk.Frame(right, bg=PANEL, padx=15, pady=13)
voice_card.pack(fill="x", pady=(0, 12))
make_label(voice_card, "VOICE ALERTS", 10, MUTED, True).pack(anchor="w")
voice_status = make_label(voice_card, "ENABLED • OFFLINE VOICE", 10, GREEN, True)
voice_status.pack(anchor="w", pady=(7, 0))

button_row = tk.Frame(right, bg=BG)
button_row.pack(fill="x", pady=(0, 12))

start_button = tk.Button(
    button_row, text="START", command=lambda: start_detection(),
    font=("Segoe UI", 10, "bold"), bg="#2563eb", fg="white",
    activebackground="#1d4ed8", activeforeground="white",
    relief="flat", padx=12, pady=10, cursor="hand2"
)
start_button.pack(side="left", fill="x", expand=True, padx=(0, 6))

stop_button = tk.Button(
    button_row, text="STOP", command=lambda: stop_detection(),
    font=("Segoe UI", 10, "bold"), bg="#334155", fg="white",
    activebackground="#475569", activeforeground="white",
    relief="flat", padx=12, pady=10, cursor="hand2", state="disabled"
)
stop_button.pack(side="left", fill="x", expand=True, padx=(6, 0))

note = make_label(
    right,
    "Prototype note: this demo recognises only classes supported by the current YOLO model. It cannot guarantee a clear or safe path.",
    9,
    MUTED,
    bg=BG,
    wraplength=275,
    justify="left"
)
note.pack(fill="x", pady=(0, 8))

bottom = tk.Frame(root, bg=BG, padx=22, pady=10)
bottom.pack(fill="x")
make_label(
    bottom,
    "ASTRIX • LOCAL PROCESSING • NO CLOUD REQUIRED AFTER SETUP",
    9,
    MUTED,
    True,
    bg=BG
).pack(side="left")
fps_label = make_label(bottom, "FPS: --", 9, MUTED, True, bg=BG)
fps_label.pack(side="right")


def initialise_resources():
    global model, camera, voice_thread
    try:
        camera_state.config(text="LOADING MODEL", fg=AMBER)
        root.update_idletasks()
        model = YOLO(MODEL_PATH)
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check camera permissions and close other apps using it."
            )
        voice_thread = threading.Thread(
            target=speak_warning_worker, daemon=True
        )
        voice_thread.start()
        camera_state.config(text="CAMERA READY", fg=GREEN)
        warning_status.config(text="READY", fg=GREEN)
        status_detail.config(text="Local model and camera are ready.")
        start_detection()
    except Exception as error:
        camera_state.config(text="NOT READY", fg=RED)
        warning_status.config(text="SETUP ERROR", fg=RED)
        status_detail.config(text=str(error))
        video_label.config(
            text="Could not start ASTRIX.\nCheck the terminal for details.",
            image=""
        )
        print("ASTRIX setup error:", error)


def start_detection():
    global running
    if running:
        return
    if model is None or camera is None or not camera.isOpened():
        warning_status.config(text="NOT READY", fg=RED)
        status_detail.config(text="Model or camera is unavailable.")
        return
    running = True
    start_button.config(state="disabled")
    stop_button.config(state="normal")
    camera_state.config(text="LIVE", fg=GREEN)
    status_detail.config(text="Monitoring the camera feed.")
    update_frame()


def stop_detection():
    global running, after_id, warning_active
    running = False
    warning_active = False
    if after_id is not None:
        try:
            root.after_cancel(after_id)
        except tk.TclError:
            pass
        after_id = None
    start_button.config(state="normal")
    stop_button.config(state="disabled")
    camera_state.config(text="STOPPED", fg=AMBER)
    warning_status.config(text="MONITORING STOPPED", fg=AMBER)
    status_detail.config(text="Camera processing is paused.")
    fps_label.config(text="FPS: --")


def update_frame():
    global after_id, photo_ref, last_warning_time, warning_active
    global latest_objects, frame_count, fps_value, fps_start

    if not running:
        return

    success, frame = camera.read()
    if not success:
        warning_status.config(text="CAMERA ERROR", fg=RED)
        status_detail.config(text="Could not read a camera frame.")
        stop_detection()
        return

    height, width = frame.shape[:2]
    zone_x1 = int(width * ZONE_LEFT)
    zone_x2 = int(width * ZONE_RIGHT)

    try:
        result = model.predict(
            source=frame,
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )[0]
    except Exception as error:
        warning_status.config(text="DETECTION ERROR", fg=RED)
        status_detail.config(text=str(error))
        print("Detection error:", error)
        stop_detection()
        return

    detected = []
    in_zone = []

    # Draw the monitored centre zone.
    overlay = frame.copy()
    cv2.rectangle(overlay, (zone_x1, 0), (zone_x2, height), (30, 120, 60), -1)
    frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
    cv2.rectangle(frame, (zone_x1, 0), (zone_x2, height), (80, 210, 130), 2)
    cv2.putText(
        frame, "MONITORED ZONE", (zone_x1 + 8, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 235, 150), 2
    )

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        center_x = (x1 + x2) // 2
        is_in_zone = zone_x1 <= center_x <= zone_x2

        detected.append((class_name, confidence, is_in_zone))
        if is_in_zone:
            in_zone.append(class_name)

        color = (80, 210, 130) if is_in_zone else (230, 180, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{class_name} {confidence:.0%}"
        cv2.rectangle(
            frame, (x1, max(0, y1 - 28)),
            (min(width - 1, x1 + max(130, len(caption) * 10)), y1),
            color, -1
        )
        cv2.putText(
            frame, caption, (x1 + 4, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 18, 28), 2
        )

    detected.sort(key=lambda item: item[1], reverse=True)
    latest_objects = detected

    objects_list.delete(0, tk.END)
    for name, confidence, is_in_zone in detected[:30]:
        zone_mark = "ZONE" if is_in_zone else "OTHER"
        objects_list.insert(
            tk.END, f"{name[:18]:18} {confidence:>5.0%}  {zone_mark}"
        )
    object_count.config(
        text=f"{len(detected)} object{'s' if len(detected) != 1 else ''}"
    )

    now = time.monotonic()
    warning_active = bool(in_zone)

    if warning_active:
        unique_names = list(dict.fromkeys(in_zone))
        warning_status.config(text="POTENTIAL OBSTACLE", fg=RED)
        status_detail.config(
            text="Recognised object centre is inside the monitored zone: "
            + ", ".join(unique_names[:4])
        )
        if now - last_warning_time >= WARNING_REPEAT_SECONDS:
            queue_warning("Warning! Obstacle detected ahead.")
            last_warning_time = now
    else:
        warning_status.config(text="NO OBJECT IN ZONE", fg=GREEN)
        status_detail.config(
            text="No recognised object centre is currently inside the monitored zone."
        )

    # Keep preview at a useful size while preserving aspect ratio.
    max_width, max_height = 800, 520
    scale = min(max_width / width, max_height / height)
    display_width = max(1, int(width * scale))
    display_height = max(1, int(height * scale))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (display_width, display_height), Image.Resampling.LANCZOS
    )
    photo_ref = ImageTk.PhotoImage(image=image)
    video_label.config(image=photo_ref, text="")

    frame_count += 1
    elapsed = now - fps_start
    if elapsed >= 1.0:
        fps_value = frame_count / elapsed
        frame_count = 0
        fps_start = now
        fps_label.config(text=f"FPS: {fps_value:.1f}")

    after_id = root.after(10, update_frame)


def on_close():
    global running, camera
    stop_detection()
    if camera is not None:
        camera.release()
    speech_stop.set()
    try:
        speech_queue.put_nowait(None)
    except queue.Full:
        pass
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.after(100, initialise_resources)
root.mainloop()
