// Variables Globales
let holistic;
let camera;
let model;
let labels = [];
let recordedData = []; // Almacena todas las grabaciones en memoria
let isRecording = false;
let currentLabel = "";

const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const statusDiv = document.getElementById('statusText');
const spellerDiv = document.getElementById('spellerText');

// Configuración de MediaPipe Holistic (con puntos y líneas finas)
function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    // Dibujar Landmarks de las manos con trazos finos
    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 1.5 });
        drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#FF0000', lineWidth: 1, radius: 2 });
    }
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 1.5 });
        drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#FF0000', lineWidth: 1, radius: 2 });
    }
    canvasCtx.restore();

    // Captura de datos durante grabación
    if (isRecording) {
        let hand = results.rightHandLandmarks || results.leftHandLandmarks;
        if (hand) {
            let frame = [];
            hand.forEach(lm => {
                frame.push(lm.x, lm.y, lm.z);
            });
            if (frame.length === 63) {
                recordedData[recordedData.length - 1].frames.push(frame);
            }
        }
    }

    // Predicción en vivo si el modelo ya está cargado o entrenado
    if (model && !isRecording) {
        let hand = results.rightHandLandmarks || results.leftHandLandmarks;
        if (hand) {
            let frame = [];
            hand.forEach(lm => frame.push(lm.x, lm.y, lm.z));
            if (frame.length === 63) {
                tf.tidy(() => {
                    const input = tf.tensor2d([frame]);
                    const prediction = model.predict(input);
                    const scores = prediction.dataSync();
                    const maxScoreIndex = prediction.argMax(1).dataSync()[0];
                    
                    if (scores[maxScoreIndex] > 0.75) {
                        const detected = labels[maxScoreIndex];
                        if (statusDiv) statusDiv.innerText = `Seña detectada: ${detected.toUpperCase()}`;
                    }
                });
            }
        }
    }
}

// Inicializar MediaPipe
holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

holistic.onResults(onResults);

// Inicializar Cámara
camera = new Camera(videoElement, {
    onFrame: async () => {
        await holistic.send({ image: videoElement });
    },
    width: 640,
    height: 480
});
camera.start();

// Eventos de Botones de Grabación
document.getElementById('btnRecord').addEventListener('click', () => {
    const labelInput = document.getElementById('labelInput').value.trim();
    if (!labelInput) {
        alert("Escribe el nombre de la seña primero (ej: Hola)");
        return;
    }

    currentLabel = labelInput.toLowerCase();
    
    recordedData.push({
        label: currentLabel,
        frames: []
    });

    isRecording = true;
    let secondsLeft = 10;
    if (statusDiv) statusDiv.innerText = `🔴 Grabando "${currentLabel}"... ${secondsLeft}s`;

    const timer = setInterval(() => {
        secondsLeft--;
        if (secondsLeft > 0) {
            if (statusDiv) statusDiv.innerText = `🔴 Grabando "${currentLabel}"... ${secondsLeft}s`;
        } else {
            clearInterval(timer);
            isRecording = false;
            const totalFrames = recordedData[recordedData.length - 1].frames.length;
            if (statusDiv) statusDiv.innerText = `¡Grabación lista! Capturados ${totalFrames} fotogramas. Total señas en memoria: ${recordedData.length}`;
            alert(`Grabación finalizada para "${currentLabel}". Fotogramas capturados: ${totalFrames}`);
        }
    }, 1000);
});

// Función para Entrenar la IA directamente
async function trainDirectly() {
    if (recordedData.length === 0) {
        alert("Primero debes hacer al menos 1 grabación de una seña.");
        return;
    }

    if (statusDiv) statusDiv.innerText = "Procesando señas guardadas...";

    try {
        let inputs = [];
        let outputs = [];
        labels = [...new Set(recordedData.map(item => item.label))];

        recordedData.forEach(item => {
            const labelIndex = labels.indexOf(item.label);
            item.frames.forEach(frame => {
                if (frame.length === 63) {
                    inputs.push(frame);
                    outputs.push(labelIndex);
                }
            });
        });

        if (inputs.length === 0) {
            alert("No se capturaron fotogramas válidos. Intenta grabar de nuevo asegurándote de mostrar la mano.");
            return;
        }

        const xs = tf.tensor2d(inputs);
        const ys = tf.oneHot(tf.tensor1d(outputs, 'int32'), labels.length);

        model = tf.sequential();
        model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [63] }));
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));

        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        if (statusDiv) statusDiv.innerText = "Entrenando IA... Por favor espera unos segundos.";

        await model.fit(xs, ys, {
            epochs: 30,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (statusDiv) statusDiv.innerText = `Entrenando IA... Época ${epoch + 1}/30 - Precisión: ${Math.round(logs.acc * 100)}%`;
                }
            }
        });

        xs.dispose();
        ys.dispose();

        if (statusDiv) statusDiv.innerText = `¡IA Entrenada con éxito! Ya puedes hacer las señas en vivo.`;
        alert(`¡Entrenamiento completado! Señas aprendidas: ${labels.join(', ')}`);

    } catch (err) {
        console.error(err);
        alert("Error al entrenar la IA: " + err.message);
    }
}

// Botones de voz y borrar
document.getElementById('btnSpeak').addEventListener('click', () => {
    const text = statusDiv ? statusDiv.innerText.replace('Seña detectada: ', '') : "";
    if (text) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'es-CL';
        window.speechSynthesis.speak(utterance);
    }
});

document.getElementById('btnClear').addEventListener('click', () => {
    if (spellerDiv) spellerDiv.innerText = "";
    if (statusDiv) statusDiv.innerText = "Texto borrado.";
});

// Guardar el modelo en la tablet
document.getElementById('btnSaveModel').addEventListener('click', async () => {
    if (!model) {
        alert("Primero debes entrenar la IA antes de poder guardarla.");
        return;
    }
    await model.save('downloads://modelo-lsch');
    localStorage.setItem('lsch_labels', JSON.stringify(labels));
    alert("¡Modelo guardado exitosamente en tus descargas!");
});

// Cargar el modelo guardado desde la tablet
document.getElementById('btnLoadModel').addEventListener('click', () => {
    document.getElementById('loadModelInput').click();
});

document.getElementById('loadModelInput').addEventListener('change', async (event) => {
    const files = event.target.files;
    if (files.length < 2) {
        alert("Debes seleccionar AMBOS archivos descargados (.json y .bin) al mismo tiempo.");
        return;
    }

    try {
        if (statusDiv) statusDiv.innerText = "Cargando modelo guardado...";
        
        model = await tf.loadLayersModel(tf.io.browserFiles([files[0], files[1]]));
        
        const savedLabels = localStorage.getItem('lsch_labels');
        if (savedLabels) {
            labels = JSON.parse(savedLabels);
        }

        if (statusDiv) statusDiv.innerText = "¡Modelo cargado con éxito! Listo para reconocer señas.";
        alert("¡IA restaurada! Señas listas para reconocer.");
    } catch (err) {
        console.error(err);
        alert("Error al cargar el modelo: " + err.message);
    }
});