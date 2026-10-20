
import pyttsx3
import time

print("Starting repeated voice test...")

for i in range(5):
    print(f"Preparing warning {i + 1}...")

    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    engine.say("Warning! Obstacle detected ahead.")
    engine.runAndWait()
    engine.stop()
    del engine

    print(f"Warning {i + 1} completed.")
    time.sleep(2)

print("Test finished.")
