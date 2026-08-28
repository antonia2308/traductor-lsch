// Variables globales
let holistic;
let camera;
let model;
let labels = [];

const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

let isRecording = false;
let recordedFrames = [];
let dataset = [];

// Elementos de la interfaz
const labelInput = document.getElementById('labelInput');
const btnRecord = document.getElementById('btnRecord');
const btnCopyData = document.getElementById('btnCopyData');
const statusDiv = document.getElementById('statusText') || document.getElementById('status');

// Inicializar MediaPipe
function initMediaPipe() {
    holistic = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    holistic.setOptions({
        modelComplexity: 0,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    holistic.onResults(onResults);

    camera = new Camera(videoElement, {
        onFrame: async () => {
            await holistic.send({ image: videoElement });
        },
        width: 640,
        height: 480
    });
    camera.start();
    if (statusDiv) statusDiv.innerText = "Cámara lista. Carga tu archivo JSON para entrenar.";
}

// Extraer puntos clave de la mano
function extractHandFeatures(results) {
    let hand = results.rightHandLandmarks || results.leftHandLandmarks;
    if (!hand) return null;
    
    let wrist = hand[0];
    let features = [];
    for (let pt of hand) {
        features.push(pt.x - wrist.x);
        features.push(pt.y - wrist.y);
        features.push(pt.z - wrist.z);
    }
    return features;
}

// Dibujar y realizar predicciones
async function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 1.5});
        drawLandmarks(canvasCtx, results.rightHandLandmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
    }
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 1.5});
        drawLandmarks(canvasCtx, results.leftHandLandmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
    }
    canvasCtx.restore();

    if (isRecording) {
        let features = extractHandFeatures(results);
        if (features) recordedFrames.push(features);
    }

    // Predicción en vivo si el modelo ya fue entrenado
    if (model && !isRecording) {
        let features = extractHandFeatures(results);
        if (features) {
            tf.tidy(() => {
                const inputTensor = tf.tensor2d([features]);
                const prediction = model.predict(inputTensor);
                const scores = prediction.dataSync();
                const maxScoreIndex = prediction.argMax(1).dataSync()[0];
                
                if (scores[maxScoreIndex] > 0.7) {
                    const detectedLabel = labels[maxScoreIndex];
                    if (statusDiv) statusDiv.innerText = `Seña detectada: ${detectedLabel.toUpperCase()} (${Math.round(scores[maxScoreIndex]*100)}%)`;
                }
            });
        }
    }
}

// Función para cargar JSON y entrenar la Red Neuronal
async function loadAndTrain(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (statusDiv) statusDiv.innerText = "Leyendo archivo de datos...";

    const reader = new FileReader();
    reader.onload = async (e) => {
        try {
            const rawData = JSON.parse(e.target.result);
            if (statusDiv) statusDiv.innerText = "Procesando datos para la IA...";

            let inputs = [];
            let outputs = [];
            labels = [...new Set(rawData.map(item => item.label))];

            rawData.forEach(item => {
                const labelIndex = labels.indexOf(item.label);
                item.frames.forEach(frame => {
                    if (frame.length === 63) {
                        inputs.push(frame);
                        outputs.push(labelIndex);
                    }
                });
            });

            if (inputs.length === 0) {
                alert("El archivo no contiene fotogramas válidos.");
                return;
            }

            const xs = tf.tensor2d(inputs);
            const ys = tf.oneHot(tf.tensor1d(outputs, 'int32'), labels.length);

            // Crear modelo de Red Neuronal
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

            if (statusDiv) statusDiv.innerText = `¡Entrenamiento completo! Muestra tu mano a la cámara haciendo "Hola".`;
            alert("¡IA entrenada con éxito! Ahora puedes hacer la seña frente a la cámara.");

        } catch (err) {
            console.error(err);
            alert("Error al procesar el archivo JSON: " + err.message);
        }
    };
    reader.readAsText(file);
}

// Eventos de botones
if (btnRecord) {
    btnRecord.addEventListener('click', () => {
        const label = labelInput ? labelInput.value.trim().toLowerCase() : '';
        if (!label) {
            alert("Por favor, ingresa el nombre de la seña.");
            return;
        }

        recordedFrames = [];
        isRecording = true;
        btnRecord.disabled = true;
        
        let countdown = 10;
        if (statusDiv) statusDiv.innerText = `Grabando "${label}"... (${countdown}s)`;

        let timer = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                if (statusDiv) statusDiv.innerText = `Grabando "${label}"... (${countdown}s)`;
            } else {
                clearInterval(timer);
                isRecording = false;
                btnRecord.disabled = false;

                if (recordedFrames.length > 0) {
                    dataset.push({ label: label, frames: recordedFrames });
                    if (statusDiv) statusDiv.innerText = `¡Grabación lista! Capturados ${recordedFrames.length} fotogramas. Total señas: ${dataset.length}`;
                }
            }
        }, 1000);
    });
}

if (btnCopyData) {
    btnCopyData.addEventListener('click', () => {
        if (dataset.length === 0) {
            alert("No hay datos grabados.");
            return;
        }
        navigator.clipboard.writeText(JSON.stringify(dataset, null, 2)).then(() => {
            alert("¡Datos copiados al portapapeles!");
        });
    });
}

window.onload = initMediaPipe;

// Función para entrenar pegando el texto directamente
async function trainFromText() {
    const rawText = document.getElementById('jsonPasteInput').value.trim();
    if (!rawText) {
        alert("Por favor pega el texto grabado en el cuadro blanco primero.");
        return;
    }

    if (statusDiv) statusDiv.innerText = "Procesando datos pegados...";

    try {
        const rawData = JSON.parse(rawText);
        let inputs = [];
        let outputs = [];
        labels = [...new Set(rawData.map(item => item.label))];

        rawData.forEach(item => {
            const labelIndex = labels.indexOf(item.label);
            item.frames.forEach(frame => {
                if (frame.length === 63) {
                    inputs.push(frame);
                    outputs.push(labelIndex);
                }
            });
        });

        if (inputs.length === 0) {
            alert("No se encontraron fotogramas válidos en el texto.");
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

        if (statusDiv) statusDiv.innerText = `¡Entrenamiento completo! Haz la seña frente a la cámara.`;
        alert("¡IA entrenada con éxito! Ahora prueba hacer la seña.");

    } catch (err) {
        console.error(err);
        alert("El texto pegado sigue estando incompleto. Asegúrate de copiar desde el inicio '[' hasta el final ']' en Google Docs.");
    }
}