import pyttsx3

for voice_name in ["David", "Zira"]:
    engine = pyttsx3.init()

    for voice in engine.getProperty("voices"):
        if voice_name in voice.name:
            engine.setProperty("voice", voice.id)
            break

    engine.say(f"ASTRIX warning. Obstacle detected ahead. Voice {voice_name}.")
    engine.runAndWait()
    engine.stop()