
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusText = document.getElementById("status");
const results = document.getElementById("results");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");

// Remove any Test Voice button left by the previous version.
document.getElementById("testVoice")?.remove();
document.getElementById("testSound")?.remove();

let model = null;
let cameraStream = null;
let running = false;
let detecting = false;
let animationId = null;
let audioContext = null;
let lastSpoken = {};
let speechQueue = [];
let speaking = false;
let selectedVoice = null;

const CONFIDENCE_THRESHOLD = 0.45;
const REPEAT_DELAY = 5000;

const ALERT_OBJECTS = new Set([
  "person", "chair", "bicycle", "motorcycle", "car",
  "bus", "truck", "dog", "bench", "potted plant",
  "backpack", "suitcase", "umbrella", "bottle",
  "dining table", "handbag"
]);

function loadVoices() {
  if (!("speechSynthesis" in window)) return;

  const voices = speechSynthesis.getVoices();

  selectedVoice =
    voices.find(v => v.lang.toLowerCase() === "en-in") ||
    voices.find(v => v.lang.toLowerCase().startsWith("en-")) ||
    voices.find(v => v.lang.toLowerCase().startsWith("en")) ||
    null;
}

if ("speechSynthesis" in window) {
  loadVoices();
  speechSynthesis.addEventListener("voiceschanged", loadVoices);
}

function speakText(text) {
  return new Promise(resolve => {
    if (!("speechSynthesis" in window)) {
      statusText.textContent =
        "Speech is not supported by this browser.";
      resolve(false);
      return;
    }

    loadVoices();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = selectedVoice?.lang || "en-IN";
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;

    if (selectedVoice) utterance.voice = selectedVoice;

    let completed = false;

    function finish(success) {
      if (completed) return;
      completed = true;
      resolve(success);
    }

    utterance.onend = () => finish(true);

    utterance.onerror = event => {
      console.error("Speech error:", event.error);
      statusText.textContent =
        "Voice warning failed: " + event.error;
      finish(false);
    };

    try {
      speechSynthesis.speak(utterance);
    } catch (error) {
      console.error(error);
      statusText.textContent = "Unable to start voice warning.";
      finish(false);
    }
  });
}

async function loadModel() {
  try {
    statusText.textContent = "Loading AI model...";
    model = await cocoSsd.load();

    statusText.textContent = "AI ready. Tap Start Detection.";
    startButton.disabled = false;
  } catch (error) {
    console.error(error);
    statusText.textContent =
      "AI model failed to load. Check internet and refresh.";
  }
}

async function startDetection() {
  if (!model || running) return;

  try {
    const AudioContextClass =
      window.AudioContext || window.webkitAudioContext;

    if (AudioContextClass) {
      if (!audioContext || audioContext.state === "closed") {
        audioContext = new AudioContextClass();
      }
      await audioContext.resume();
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera requires HTTPS or localhost.");
    }

    statusText.textContent = "Opening camera...";

    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" } }
    });

    video.srcObject = cameraStream;
    await video.play();

    // Resume voice engine after the Start button interaction.
    if ("speechSynthesis" in window) {
      loadVoices();
      speechSynthesis.cancel();
    }

    running = true;
    startButton.disabled = true;
    stopButton.disabled = false;

    statusText.textContent = "Camera active. Detecting objects...";
    detectLoop();
  } catch (error) {
    console.error(error);
    statusText.textContent = "Could not start: " + error.message;
    stopCamera();
  }
}

function drawBoxes(predictions) {
  const width = video.videoWidth;
  const height = video.videoHeight;

  if (!width || !height) return [];

  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }

  ctx.clearRect(0, 0, width, height);

  // Cyan centre warning zone.
  const zoneWidth = width * 0.30;
  const zoneX = (width - zoneWidth) / 2;

  ctx.strokeStyle = "#00ffff";
  ctx.lineWidth = Math.max(3, width / 240);
  ctx.strokeRect(zoneX, 0, zoneWidth, height);

  const inZone = predictions.filter(item => {
    const [x, , boxWidth] = item.bbox;
    const centreX = x + boxWidth / 2;
    return centreX >= zoneX && centreX <= zoneX + zoneWidth;
  });

  predictions.forEach(item => {
    const [x, y, boxWidth, boxHeight] = item.bbox;
    const inside = inZone.includes(item);
    const label =
      `${item.class} ${Math.round(item.score * 100)}%`;
    const fontSize = Math.max(16, width / 30);
    const lineHeight = fontSize + 8;

    ctx.font = `${fontSize}px Arial`;
    ctx.lineWidth = Math.max(3, width / 240);
    ctx.strokeStyle = inside ? "#ff2222" : "#246bff";
    ctx.strokeRect(x, y, boxWidth, boxHeight);

    const labelWidth = ctx.measureText(label).width + 10;
    const labelY = Math.max(lineHeight, y);

    ctx.fillStyle = inside ? "#ff2222" : "#246bff";
    ctx.fillRect(x, labelY - lineHeight, labelWidth, lineHeight);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 5, labelY - 5);
  });

  return inZone;
}

function updateResults(predictions, inZone) {
  results.replaceChildren();

  if (predictions.length === 0) {
    const line = document.createElement("div");
    line.textContent = "No common objects detected.";
    results.appendChild(line);
    return;
  }

  predictions.forEach(item => {
    const line = document.createElement("div");
    line.textContent =
      `${inZone.includes(item) ? "WARNING ZONE: " : ""}` +
      `${item.class} — ${Math.round(item.score * 100)}% confidence`;
    results.appendChild(line);
  });
}

function playWarningBeep() {
  if (!audioContext || audioContext.state !== "running") return;

  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();

  oscillator.type = "sine";
  oscillator.frequency.value = 880;
  gain.gain.value = 0.12;

  oscillator.connect(gain);
  gain.connect(audioContext.destination);

  oscillator.start();
  oscillator.stop(audioContext.currentTime + 0.18);
}

function queueAnnouncements(inZone) {
  const now = Date.now();

  for (const item of inZone) {
    const name = item.class;

    if (!ALERT_OBJECTS.has(name)) continue;
    if (now - (lastSpoken[name] || 0) < REPEAT_DELAY) continue;
    if (speechQueue.includes(name)) continue;

    lastSpoken[name] = now;
    speechQueue.push(name);
  }

  processSpeechQueue();
}

async function processSpeechQueue() {
  if (!running || speaking || speechQueue.length === 0) return;

  speaking = true;

  try {
    while (running && speechQueue.length > 0) {
      const name = speechQueue.shift();

      statusText.textContent = `Speaking warning: ${name}`;

      playWarningBeep();

      const success = await speakText(
        `Warning! ${name} ahead!`
      );

      if (!success) {
        // Stop retrying continuously if the browser reports an error.
        speechQueue = [];
        break;
      }
    }
  } finally {
    speaking = false;

    if (running && speechQueue.length > 0) {
      processSpeechQueue();
    }
  }
}

async function detectLoop() {
  if (!running || detecting) return;

  if (video.readyState < 2 || !model) {
    animationId = requestAnimationFrame(detectLoop);
    return;
  }

  detecting = true;

  try {
    const predictions = await model.detect(video);

    const visible = predictions.filter(
      item => item.score >= CONFIDENCE_THRESHOLD
    );

    const inZone = drawBoxes(visible);
    updateResults(visible, inZone);

    if (inZone.length > 0) {
      statusText.textContent =
        "Object detected inside the centre zone.";
      queueAnnouncements(inZone);
    } else {
      statusText.textContent =
        "Camera active. No detected object in the centre zone.";
    }
  } catch (error) {
    console.error("Detection error:", error);
    statusText.textContent = "Detection error. Please refresh.";
  } finally {
    detecting = false;

    if (running) {
      animationId = requestAnimationFrame(detectLoop);
    }
  }
}

function stopCamera() {
  running = false;

  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }

  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  detecting = false;
  speechQueue = [];
  speaking = false;

  if ("speechSynthesis" in window) {
    speechSynthesis.cancel();
  }

  if (audioContext && audioContext.state !== "closed") {
    audioContext.close();
  }
  audioContext = null;

  startButton.disabled = !model;
  stopButton.disabled = true;
  statusText.textContent = "Detection stopped. Camera is off.";
}

startButton.addEventListener("click", startDetection);
stopButton.addEventListener("click", stopCamera);
window.addEventListener("pagehide", stopCamera);

startButton.disabled = true;
stopButton.disabled = true;

loadModel();
