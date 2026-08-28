// Variables globales
let holistic;
let camera;
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

let isRecording = false;
let recordedFrames = [];
let dataset = [];

// Elementos de la interfaz
const labelInput = document.getElementById('labelInput');
const btnRecord = document.getElementById('btnRecord');
const btnTrain = document.getElementById('btnTrain');
const btnPredict = document.getElementById('btnPredict');
const btnCopyData = document.getElementById('btnCopyData');
const statusDiv = document.getElementById('status');

// Inicializar MediaPipe Holistic optimizado para tablet
function initMediaPipe() {
    holistic = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    // modelComplexity: 0 hace que procese mucho más rápido en dispositivos móviles
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
    if (statusDiv) statusDiv.innerText = "Estado: Cámara lista. Lista para grabar o predecir.";
}

// Extraer los 21 puntos clave de la mano derecha o izquierda
function extractHandFeatures(results) {
    let hand = results.rightHandLandmarks || results.leftHandLandmarks;
    if (!hand) return null;
    
    // Normalizar coordenadas respecto a la muñeca (punto 0)
    let wrist = hand[0];
    let features = [];
    for (let pt of hand) {
        features.push(pt.x - wrist.x);
        features.push(pt.y - wrist.y);
        features.push(pt.z - wrist.z);
    }
    return features;
}

// Procesar cada fotograma de la cámara con puntos más finos
function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    // Dibujar manos con puntos delgados (radius: 2)
    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 1.5});
        drawLandmarks(canvasCtx, results.rightHandLandmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
    }
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 1.5});
        drawLandmarks(canvasCtx, results.leftHandLandmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
    }
    canvasCtx.restore();

    // Captura de datos durante la grabación
    if (isRecording) {
        let features = extractHandFeatures(results);
        if (features) {
            recordedFrames.push(features);
        }
    }
}

// Lógica para Grabar Muestras (10 Segundos)
if (btnRecord) {
    btnRecord.addEventListener('click', () => {
        const label = labelInput ? labelInput.value.trim().toLowerCase() : '';
        if (!label) {
            alert("Por favor, ingresa el nombre de la seña (ejemplo: hola)");
            return;
        }

        recordedFrames = [];
        isRecording = true;
        btnRecord.disabled = true;
        
        let countdown = 10;
        if (statusDiv) statusDiv.innerText = `Grabando seña "${label}"... Repite el gesto continuamente! (${countdown}s)`;

        let timer = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                if (statusDiv) statusDiv.innerText = `Grabando seña "${label}"... Repite el gesto continuamente! (${countdown}s)`;
            } else {
                clearInterval(timer);
                isRecording = false;
                btnRecord.disabled = false;

                if (recordedFrames.length > 0) {
                    dataset.push({ label: label, frames: recordedFrames });
                    if (statusDiv) statusDiv.innerText = `¡Grabación finalizada! Capturados ${recordedFrames.length} fotogramas para "${label}". Muestras totales: ${dataset.length}`;
                } else {
                    if (statusDiv) statusDiv.innerText = "No se detectó ninguna mano durante la grabación. Intenta de nuevo.";
                }
            }
        }, 1000);
    });
}

// Botón para Copiar Datos al Portapapeles (Solución Tablet)
if (btnCopyData) {
    btnCopyData.addEventListener('click', () => {
        if (dataset.length === 0) {
            alert("No hay muestras grabadas aún.");
            return;
        }
        const dataStr = JSON.stringify(dataset, null, 2);
        navigator.clipboard.writeText(dataStr).then(() => {
            alert("¡Datos copiados al portapapeles con éxito! Puedes pegarlos donde quieras.");
        }).catch(err => {
            console.error("Error al copiar: ", err);
            alert("No se pudo copiar automáticamente.");
        });
    });
}

// Inicializar al cargar la página
window.onload = initMediaPipe;