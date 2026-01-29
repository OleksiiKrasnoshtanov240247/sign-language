// WebSocket connection
let ws = null;
let sessionId = null;
let isRunning = false;
let stream = null;
let captureInterval = null;

// DOM elements
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const targetLetterElement = document.getElementById('targetLetter');
const predictionLetterElement = document.getElementById('predictionLetter');
const confidenceElement = document.getElementById('confidence');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const matchDots = document.getElementById('matchDots');
const correctCountElement = document.getElementById('correctCount');
const totalCountElement = document.getElementById('totalCount');
const accuracyElement = document.getElementById('accuracy');
const timeDisplay = document.getElementById('timeDisplay');
const hintMessage = document.getElementById('hintMessage');
const hintText = document.getElementById('hintText');
const successMessage = document.getElementById('successMessage');
const timeoutMessage = document.getElementById('timeoutMessage');

// Initialize
function init() {
    console.log('Initializing app...');
    console.log('Start button:', startBtn);
    console.log('Stop button:', stopBtn);
    console.log('Webcam element:', webcamElement);
    
    startBtn.addEventListener('click', startDetection);
    stopBtn.addEventListener('click', stopDetection);
    
    console.log('App initialized successfully');
}

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        statusText.textContent = 'Connected';
        statusBadge.classList.add('connected');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerResponse(data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        statusText.textContent = 'Connection error';
        statusBadge.classList.remove('connected');
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        statusText.textContent = 'Disconnected';
        statusBadge.classList.remove('connected');
        
        if (isRunning) {
            // Try to reconnect
            setTimeout(connectWebSocket, 2000);
        }
    };
}

// Start detection
async function startDetection() {
    console.log('Starting detection...');
    try {
        // Request webcam
        console.log('Requesting webcam access...');
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        
        console.log('Webcam access granted');
        webcamElement.srcObject = stream;
        
        // Connect WebSocket
        console.log('Connecting to WebSocket...');
        connectWebSocket();
        
        // Start capture loop
        setTimeout(() => {
            console.log('Starting capture loop...');
            startCaptureLoop();
        }, 1000);
        
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusText.textContent = 'Running';
        console.log('Detection started successfully');
        
    } catch (error) {
        console.error('Error starting detection:', error);
        alert('Could not access webcam. Please check permissions.');
    }
}

// Stop detection
function stopDetection() {
    isRunning = false;
    
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusText.textContent = 'Stopped';
    statusBadge.classList.remove('connected', 'hand-detected');
}

// Capture and send frames
function startCaptureLoop() {
    captureInterval = setInterval(() => {
        if (!isRunning || !ws || ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        const frameData = captureFrame();
        if (frameData) {
            const message = {
                frame: frameData,
                session_id: sessionId
            };
            ws.send(JSON.stringify(message));
        }
    }, 200); // Send frame every 200ms (5 FPS)
}

// Capture frame from video
function captureFrame() {
    if (!webcamElement.videoWidth) return null;
    
    canvasElement.width = webcamElement.videoWidth;
    canvasElement.height = webcamElement.videoHeight;
    
    const ctx = canvasElement.getContext('2d');
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(webcamElement, -canvasElement.width, 0, canvasElement.width, canvasElement.height);
    ctx.restore();
    
    return canvasElement.toDataURL('image/jpeg', 0.8);
}

// Handle server response
function handleServerResponse(data) {
    // Update session ID
    if (data.session_id) {
        sessionId = data.session_id;
    }
    
    // Update hand detection status
    if (data.hand_detected) {
        statusBadge.classList.add('hand-detected');
    } else {
        statusBadge.classList.remove('hand-detected');
    }
    
    // Update target letter
    if (data.current_letter) {
        targetLetterElement.textContent = data.current_letter;
    }
    
    // Update prediction
    if (data.prediction) {
        predictionLetterElement.textContent = data.prediction.predicted_class;
        const conf = (data.prediction.confidence * 100).toFixed(0);
        confidenceElement.textContent = `${conf}%`;
    } else {
        predictionLetterElement.textContent = '-';
        confidenceElement.textContent = '-';
    }
    
    // Update match indicator
    updateMatchDots(data.consecutive_matches || 0);
    
    // Update stats
    if (data.progress) {
        correctCountElement.textContent = data.progress.total_correct;
        totalCountElement.textContent = data.progress.total_attempts;
        accuracyElement.textContent = `${data.progress.accuracy}%`;
        
        // Update timer
        const timeElapsed = data.progress.time_elapsed || 0;
        const timeLimit = data.progress.time_limit || 45;
        timeDisplay.textContent = `${timeElapsed}s / ${timeLimit}s`;
    }
    
    // Handle success
    if (data.success) {
        showSuccess();
    }
    
    // Handle timeout
    if (data.timeout) {
        showTimeout();
    }
    
    // Handle hints
    if (data.show_hint && data.hint_message) {
        showHint(data.hint_message);
    }
}

// Update match dots
function updateMatchDots(matchCount) {
    const dots = matchDots.querySelectorAll('.dot');
    dots.forEach((dot, index) => {
        if (index < matchCount) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });
}

// Show success message
function showSuccess() {
    successMessage.style.display = 'block';
    setTimeout(() => {
        successMessage.style.display = 'none';
    }, 2000);
    
    // Reset match dots
    updateMatchDots(0);
}

// Show timeout message
function showTimeout() {
    timeoutMessage.style.display = 'block';
    setTimeout(() => {
        timeoutMessage.style.display = 'none';
    }, 2000);
    
    // Reset match dots
    updateMatchDots(0);
}

// Show hint message
function showHint(message) {
    hintText.textContent = message;
    hintMessage.style.display = 'block';
    setTimeout(() => {
        hintMessage.style.display = 'none';
    }, 5000);
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
