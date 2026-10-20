
from ultralytics import YOLO
import cv2
import threading
import time
import pyttsx3

# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

print("Loading ASTRIX...")
model = YOLO("yolo11n.pt")

# ==========================================
# 2. WARNING CONTROL
# ==========================================

warning_active = threading.Event()
stop_program = threading.Event()


def speak_warning():
    """Create a fresh speech engine for each warning."""
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    try:
        engine.say("Warning! Obstacle detected ahead.")
        engine.runAndWait()
    finally:
        engine.stop()
        del engine


def voice_worker():
    """Repeat warnings while an object remains in the zone."""
    while not stop_program.is_set():

        if not warning_active.wait(timeout=0.2):
            continue

        while warning_active.is_set() and not stop_program.is_set():
            try:
                speak_warning()
            except Exception as error:
                print("Voice error:", error)
                time.sleep(1)
                continue

            # Wait 2 seconds before the next warning
            for _ in range(20):
                if stop_program.is_set() or not warning_active.is_set():
                    break
                time.sleep(0.1)


voice_thread = threading.Thread(
    target=voice_worker,
    daemon=True
)
voice_thread.start()

# ==========================================
# 3. START CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not available.")
    stop_program.set()
    voice_thread.join(timeout=3)
    raise SystemExit

print("ASTRIX running. Press Q to quit.")

# ==========================================
# 4. OBJECT DETECTION
# ==========================================

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        height, width = frame.shape[:2]

        # Central warning zone
        left = int(width * 0.35)
        right = int(width * 0.65)

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        output = results[0].plot()

        cv2.rectangle(
            output,
            (left, 0),
            (right, height),
            (255, 255, 0),
            2
        )

        object_in_zone = False

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            center_x = (x1 + x2) // 2

            if left < center_x < right:
                object_in_zone = True
                break

        if object_in_zone:
            warning_active.set()

            cv2.putText(
                output,
                "OBJECT IN WARNING ZONE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        else:
            warning_active.clear()

        cv2.imshow("ASTRIX - Voice Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    warning_active.clear()
    stop_program.set()

    camera.release()
    cv2.destroyAllWindows()

    voice_thread.join(timeout=3)

    print("ASTRIX stopped.")
