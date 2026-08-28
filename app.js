const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const statusText = document.getElementById('statusText');
const subStatus = document.getElementById('subStatus');
const spellerText = document.getElementById('spellerText');

let model = null;
let labels = [];
let isTrained = false;

const FRAMES_PER_SAMPLE = 30;
let liveBuffer = [];
let accumulatedText = "";

// 1. Cargar el JSON descargado y Entrenar la IA
async function loadAndTrain(event) {
  const file = event.target.files[0];
  if (!file) return;

  statusText.textContent = "Leyendo archivo de datos...";
  const reader = new FileReader();

  reader.onload = async (e) => {
    const rawData = JSON.parse(e.target.result);
    if (rawData.length === 0) {
      alert("El archivo no contiene muestras.");
      return;
    }

    // Extraer etiquetas únicas (Hola, Gracias, etc.)
    labels = [...new Set(rawData.map(item => item.label))];
    
    // Preparar tensores para TensorFlow.js
    const inputs = [];
    const outputs = [];

    rawData.forEach(sample => {
      // Aplanar los 30 cuadros en un solo vector continuo
      const flatFrames = sample.frames.flat();
      inputs.push(flatFrames);
      outputs.push(labels.indexOf(sample.label));
    });

    const xs = tf.tensor2d(inputs);
    const ys = tf.oneHot(tf.tensor1d(outputs, 'int32'), labels.length);

    // Crear la arquitectura de la Red Neuronal
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [inputs[0].length] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    statusText.textContent = "🧠 Entrenando IA...";
    subStatus.textContent = "Por favor espera unos segundos...";

    // Entrenar la red neuronal
    await model.fit(xs, ys, {
      epochs: 40,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          subStatus.textContent = `Progreso: Época ${epoch + 1}/40 - Precisión: ${(logs.acc * 100).toFixed(0)}%`;
        }
      }
    });

    isTrained = true;
    statusText.textContent = "¡IA Entrenada con éxito!";
    subStatus.textContent = "Ponte frente a la cámara y realiza tus señas";
  };

  reader.readAsText(file);
}

// 2. Extraer puntos del cuerpo y manos
function extractFeatures(results) {
  const featureVector = [];
  if (results.poseLandmarks) {
    [11, 12, 13, 14, 15, 16].forEach(i => {
      featureVector.push(results.poseLandmarks[i].x, results.poseLandmarks[i].y, results.poseLandmarks[i].z);
    });
  } else { featureVector.push(...Array(18).fill(0)); }

  if (results.leftHandLandmarks) {
    results.leftHandLandmarks.forEach(p => featureVector.push(p.x, p.y, p.z));
  } else { featureVector.push(...Array(63).fill(0)); }

  if (results.rightHandLandmarks) {
    results.rightHandLandmarks.forEach(p => featureVector.push(p.x, p.y, p.z));
  } else { featureVector.push(...Array(63).fill(0)); }

  return featureVector;
}

// 3. MediaPipe Holistic
const holistic = new Holistic({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});
holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.6, minTrackingConfidence: 0.6 });

holistic.onResults(results => {
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (results.poseLandmarks) drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00b894', lineWidth: 2 });
  if (results.leftHandLandmarks) drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#a29bfe', lineWidth: 2 });
  if (results.rightHandLandmarks) drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#ff7675', lineWidth: 2 });

  if (isTrained) {
    const features = extractFeatures(results);
    liveBuffer.push(features);
    if (liveBuffer.length > FRAMES_PER_SAMPLE) liveBuffer.shift();

    if (liveBuffer.length === FRAMES_PER_SAMPLE) {
      tf.tidy(() => {
        const inputTensor = tf.tensor2d([liveBuffer.flat()]);
        const prediction = model.predict(inputTensor);
        const scores = prediction.dataSync();
        const maxScoreIndex = prediction.argMax(1).dataSync()[0];
        const confidence = scores[maxScoreIndex];

        if (confidence > 0.82) {
          const detectedWord = labels[maxScoreIndex];
          statusText.textContent = `Traducción: "${detectedWord}"`;
          subStatus.textContent = `Confianza IA: ${(confidence * 100).toFixed(0)}%`;

          if (!accumulatedText.endsWith(detectedWord)) {
            accumulatedText += ` ${detectedWord}`;
            spellerText.textContent = accumulatedText.trim();
            liveBuffer = []; // Limpiar para la siguiente palabra
          }
        } else {
          statusText.textContent = "Analizando movimiento...";
          subStatus.textContent = "Realiza una seña aprendida";
        }
      });
    }
  }

  canvasCtx.restore();
});

function drawConnectors(ctx, lm, connections, style) {
  ctx.strokeStyle = style.color; ctx.lineWidth = style.lineWidth;
  for (const c of connections) {
    if (lm[c[0]] && lm[c[1]]) {
      ctx.beginPath();
      ctx.moveTo(lm[c[0]].x * canvasElement.width, lm[c[0]].y * canvasElement.height);
      ctx.lineTo(lm[c[1]].x * canvasElement.width, lm[c[1]].y * canvasElement.height);
      ctx.stroke();
    }
  }
}

const camera = new Camera(videoElement, {
  onFrame: async () => { await holistic.send({ image: videoElement }); },
  facingMode: 'user', width: 640, height: 480
});
camera.start();

document.getElementById('btnSpeak').addEventListener('click', () => {
  if (!spellerText.textContent) return;
  const u = new SpeechSynthesisUtterance(spellerText.textContent);
  u.lang = 'es-CL';
  window.speechSynthesis.speak(u);
});
document.getElementById('btnClear').addEventListener('click', () => { accumulatedText = ""; spellerText.textContent = ""; });