// Application state
let isRunning = false;
let stream = null;
let predictionInterval = null;
let currentTargetLetter = 'A';
let correctCount = 0;
let totalAttempts = 0;

// Available letters for sign language (A-Z, excluding J and Z which require motion)
const availableLetters = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('');

// DOM elements
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const targetLetterElement = document.getElementById('targetLetter');
const predictionLetterElement = document.getElementById('predictionLetter');
const confidenceElement = document.getElementById('confidence');
const matchIndicatorElement = document.getElementById('matchIndicator');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const nextLetterBtn = document.getElementById('nextLetterBtn');
const correctCountElement = document.getElementById('correctCount');
const totalCountElement = document.getElementById('totalCount');
const accuracyElement = document.getElementById('accuracy');

// Initialize
function init() {
    setupEventListeners();
    updateStats();
    setRandomTargetLetter();
    statusText.textContent = 'Ready to start';
}

// Event listeners
function setupEventListeners() {
    startBtn.addEventListener('click', startRecognition);
    stopBtn.addEventListener('click', stopRecognition);
    resetBtn.addEventListener('click', resetStats);
    nextLetterBtn.addEventListener('click', setRandomTargetLetter);
}

// Start webcam and recognition
async function startRecognition() {
    try {
        // Request webcam access
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 1280, height: 720 } 
        });
        
        webcamElement.srcObject = stream;
        
        // Update UI
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusIndicator.classList.add('active');
        statusText.textContent = 'Running';
        
        // Start prediction loop
        startPredictionLoop();
        
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Could not access webcam. Please ensure you have granted camera permissions.');
    }
}

// Stop recognition
function stopRecognition() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (predictionInterval) {
        clearInterval(predictionInterval);
        predictionInterval = null;
    }
    
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusIndicator.classList.remove('active');
    statusText.textContent = 'Stopped';
    
    // Clear prediction display
    predictionLetterElement.textContent = '-';
    confidenceElement.textContent = 'Confidence: --';
    matchIndicatorElement.textContent = '';
    matchIndicatorElement.className = 'match-indicator';
}

// Prediction loop
function startPredictionLoop() {
    // Run predictions every 500ms
    predictionInterval = setInterval(async () => {
        await getPrediction();
    }, 500);
}

// Get prediction from backend
async function getPrediction() {
    try {
        // Capture frame from video
        const frame = captureFrame();
        
        if (!frame) return;
        
        // TODO: Send frame to backend for prediction
        // For now, simulate with mock data
        const mockPrediction = getMockPrediction();
        
        // Update UI with prediction
        updatePredictionDisplay(mockPrediction);
        
        // In production, this would be:
        // const response = await fetch('/api/predict', {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify({ frame: frame })
        // });
        // const prediction = await response.json();
        // updatePredictionDisplay(prediction);
        
    } catch (error) {
        console.error('Error getting prediction:', error);
    }
}

// Capture frame from video
function captureFrame() {
    if (!webcamElement.videoWidth) return null;
    
    canvasElement.width = webcamElement.videoWidth;
    canvasElement.height = webcamElement.videoHeight;
    
    const ctx = canvasElement.getContext('2d');
    ctx.drawImage(webcamElement, 0, 0);
    
    return canvasElement.toDataURL('image/jpeg');
}

// Mock prediction (replace with actual backend call)
function getMockPrediction() {
    // Simulate prediction with random letter and confidence
    // In production, this will be replaced with actual model prediction
    const predictedLetter = availableLetters[Math.floor(Math.random() * availableLetters.length)];
    const confidence = Math.random() * 0.4 + 0.6; // 60-100%
    
    return {
        letter: predictedLetter,
        confidence: confidence
    };
}

// Update prediction display
function updatePredictionDisplay(prediction) {
    predictionLetterElement.textContent = prediction.letter;
    confidenceElement.textContent = `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`;
    
    // Check if prediction matches target
    const isMatch = prediction.letter === currentTargetLetter;
    
    if (prediction.confidence > 0.75) {
        if (isMatch) {
            matchIndicatorElement.textContent = '✓ Correct!';
            matchIndicatorElement.className = 'match-indicator correct';
        } else {
            matchIndicatorElement.textContent = '✗ Try again';
            matchIndicatorElement.className = 'match-indicator incorrect';
        }
    } else {
        matchIndicatorElement.textContent = '';
        matchIndicatorElement.className = 'match-indicator';
    }
}

// Set random target letter
function setRandomTargetLetter() {
    const previousLetter = currentTargetLetter;
    
    // Ensure we get a different letter
    do {
        currentTargetLetter = availableLetters[Math.floor(Math.random() * availableLetters.length)];
    } while (currentTargetLetter === previousLetter && availableLetters.length > 1);
    
    targetLetterElement.textContent = currentTargetLetter;
    
    // Clear previous prediction
    predictionLetterElement.textContent = '-';
    confidenceElement.textContent = 'Confidence: --';
    matchIndicatorElement.textContent = '';
    matchIndicatorElement.className = 'match-indicator';
}

// Update stats display
function updateStats() {
    correctCountElement.textContent = correctCount;
    totalCountElement.textContent = totalAttempts;
    
    if (totalAttempts > 0) {
        const accuracy = (correctCount / totalAttempts * 100).toFixed(1);
        accuracyElement.textContent = `${accuracy}%`;
    } else {
        accuracyElement.textContent = '--%';
    }
}

// Reset statistics
function resetStats() {
    correctCount = 0;
    totalAttempts = 0;
    updateStats();
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}