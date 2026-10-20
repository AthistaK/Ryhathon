
from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time
import queue

print("Loading ASTRIX...")
model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not available.")
    raise SystemExit

speech_queue = queue.Queue()
stop_event = threading.Event()


def voice_worker():
    while not stop_event.is_set():
        try:
            label = speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if label is None:
            speech_queue.task_done()
            break

        try:
            # Reinitialize for every message, as in our working test
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 1.0)
            engine.say(f"{label} detected.")
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as error:
            print("Voice error:", error)
        finally:
            speech_queue.task_done()


voice_thread = threading.Thread(
    target=voice_worker,
    daemon=True
)
voice_thread.start()

last_spoken = {}
COOLDOWN_SECONDS = 4

print("ASTRIX running. Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.20,
            verbose=False
        )

        output = results[0].plot()
        detected_now = set()
        current_time = time.monotonic()

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            detected_now.add(class_name)

            # Display class and confidence
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            cv2.putText(
                output,
                f"{class_name} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Announce any recognised class with a cooldown
        for class_name in detected_now:
            previous_time = last_spoken.get(class_name, 0)

            if current_time - previous_time >= COOLDOWN_SECONDS:
                # One announcement at a time; avoid stale messages
                if speech_queue.empty():
                    speech_queue.put(class_name)
                    last_spoken[class_name] = current_time

        cv2.imshow("ASTRIX - All Object Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

    speech_queue.put(None)
    voice_thread.join(timeout=5)

    if voice_thread.is_alive():
        stop_event.set()

    print("ASTRIX stopped.")
