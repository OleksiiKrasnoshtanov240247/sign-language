// WebSocket connection
let ws = null;
let sessionId = null;
let isRecording = false;
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
const recordBtn = document.getElementById('recordBtn');
const modeCheckbox = document.getElementById('modeCheckbox');
const tutorialGif = document.getElementById('tutorialGif');
const correctCountElement = document.getElementById('correctCount');
const totalCountElement = document.getElementById('totalCount');
const accuracyElement = document.getElementById('accuracy');
const timeDisplay = document.getElementById('timeDisplay');
const hintMessage = document.getElementById('hintMessage');
const hintText = document.getElementById('hintText');
const successMessage = document.getElementById('successMessage');
const timeoutMessage = document.getElementById('timeoutMessage');
const recordingProgress = document.getElementById('recordingProgress');

// State
let currentMode = 'sequential';  // Track current mode

// Initialize
function init() {
    console.log('Initializing app...');
    
    // Check if elements exist
    console.log('modeCheckbox:', modeCheckbox);
    console.log('Toggle container:', document.getElementById('modeSwitchContainer'));
    
    startBtn.addEventListener('click', startSession);
    recordBtn.addEventListener('click', toggleRecording);
    
    if (modeCheckbox) {
        modeCheckbox.addEventListener('change', toggleMode);
        console.log('✅ Mode toggle event listener attached');
    } else {
        console.error('❌ modeCheckbox not found!');
    }
    
    // Initially disable buttons
    recordBtn.disabled = true;
    if (modeCheckbox) {
        modeCheckbox.disabled = true;
    }
    
    // Initialize mode display
    updateModeToggle();
    
    console.log('App initialized');
}

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        updateStatus('connected', 'Connected');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerResponse(data);
    };
    
    ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        updateStatus('disconnected', 'Connection error');
    };
    
    ws.onclose = () => {
        console.log('📡 WebSocket closed');
        updateStatus('disconnected', 'Disconnected');
        // Auto-reconnect after delay
        setTimeout(() => {
            if (stream) {
                connectWebSocket();
            }
        }, 3000);
    };
}

// Toggle letter order mode
async function toggleMode() {
    console.log('🔄 toggleMode called');
    console.log('Session ID:', sessionId);
    console.log('Checkbox state:', modeCheckbox.checked);
    
    if (!sessionId) {
        console.error('No session ID');
        modeCheckbox.checked = !modeCheckbox.checked; // Revert
        return;
    }
    
    // Determine new mode based on checkbox state
    const newMode = modeCheckbox.checked ? 'random' : 'sequential';
    
    try {
        const response = await fetch(`/api/session/${sessionId}/mode`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode: newMode })
        });
        
        if (response.ok) {
            const data = await response.json();
            currentMode = data.mode;
            console.log(`✅ Mode changed to: ${currentMode}`);
        } else {
            console.error('Failed to change mode');
            // Revert checkbox on failure
            modeCheckbox.checked = !modeCheckbox.checked;
        }
    } catch (error) {
        console.error('Error changing mode:', error);
        // Revert checkbox on error
        modeCheckbox.checked = !modeCheckbox.checked;
    }
}

// Update mode toggle visual state
function updateModeToggle() {
    if (modeCheckbox) {
        modeCheckbox.checked = (currentMode === 'random');
        console.log('Mode toggle updated:', currentMode, 'checked:', modeCheckbox.checked);
    }
}

// Start session
async function startSession() {
    try {
        // Start webcam
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        webcamElement.srcObject = stream;
        
        // Connect WebSocket
        connectWebSocket();
        
        // Start sending frames
        startCaptureLoop();
        
        // Update UI
        startBtn.disabled = true;
        recordBtn.disabled = false;
        modeCheckbox.disabled = false;
        startBtn.textContent = 'Running';
        
        console.log('✅ Session started');
    } catch (error) {
        console.error('❌ Failed to start:', error);
        alert('Failed to access camera. Please check permissions.');
    }
}

// Toggle recording
function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

// Start recording
function startRecording() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        console.error('WebSocket not ready');
        return;
    }
    
    isRecording = true;
    recordBtn.textContent = 'Stop Recording';
    recordBtn.classList.add('recording');
    
    // Hide messages
    successMessage.style.display = 'none';
    timeoutMessage.style.display = 'none';
    hintMessage.style.display = 'none';
    
    // Show recording progress
    if (recordingProgress) {
        recordingProgress.style.display = 'block';
    }
    
    // Send start recording command
    ws.send(JSON.stringify({
        type: 'start_recording',
        session_id: sessionId
    }));
    
    console.log('🔴 Recording started');
}

// Stop recording
function stopRecording() {
    isRecording = false;
    recordBtn.textContent = 'Record';
    recordBtn.classList.remove('recording');
    
    // Hide recording progress
    if (recordingProgress) {
        recordingProgress.style.display = 'none';
    }
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'stop_recording',
            session_id: sessionId
        }));
    }
    
    console.log('⏹️ Recording stopped');
}

// Capture and send frames
function startCaptureLoop() {
    captureInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            captureFrame();
        }
    }, 100); // 10 FPS
}

// Capture frame from video
function captureFrame() {
    const canvas = canvasElement;
    const context = canvas.getContext('2d');
    
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    
    context.drawImage(webcamElement, 0, 0);
    
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    
    ws.send(JSON.stringify({
        type: 'frame',
        frame: frameData,
        session_id: sessionId
    }));
}

// Handle server response
function handleServerResponse(data) {
    console.log('📨 Server response:', data);
    
    // Update session ID
    if (data.session_id && !sessionId) {
        sessionId = data.session_id;
        console.log('Session ID:', sessionId);
    }
    
    // Update progress
    if (data.progress) {
        const prog = data.progress;
        targetLetterElement.textContent = prog.current_letter || data.current_letter;
        correctCountElement.textContent = prog.total_correct || 0;
        totalCountElement.textContent = prog.total_attempts || 0;
        
        const accuracy = prog.accuracy || 0;
        accuracyElement.textContent = accuracy.toFixed(1) + '%';
        
        // Update timer
        if (prog.time_remaining !== undefined) {
            timeDisplay.textContent = Math.ceil(prog.time_remaining) + 's';
        }
        
        // Update tutorial GIF
        if (prog.tutorial_url) {
            tutorialGif.src = prog.tutorial_url;
            tutorialGif.style.display = 'block';
        }
        
        // Update mode if provided
        if (prog.mode && prog.mode !== currentMode) {
            currentMode = prog.mode;
            updateModeToggle();
        }
    }
    
    // Update hand detection status
    if (data.hand_detected !== undefined) {
        if (data.hand_detected) {
            updateStatus('hand-detected', 'Hand Detected');
        } else {
            updateStatus('connected', data.message || 'No Hand');
        }
    }
    
    // Update recording status
    if (data.recording !== undefined) {
        if (data.recording) {
            const message = data.message || 'Recording...';
            statusText.textContent = message;
            
            // Show buffer progress for dynamic letters
            if (data.buffer_progress !== undefined && recordingProgress) {
                const pct = (data.buffer_progress * 100).toFixed(0);
                recordingProgress.textContent = `${pct}%`;
            }
        } else {
            // Recording finished
            isRecording = false;
            recordBtn.textContent = 'Record';
            recordBtn.classList.remove('recording');
            
            if (recordingProgress) {
                recordingProgress.style.display = 'none';
            }
        }
    }
    
    // Update prediction
    if (data.prediction) {
        const pred = data.prediction;
        const predictedClass = pred.predicted_class || pred.letter || '-';
        // Replace "Nonsense" with a dash for better UI
        const displayLetter = predictedClass === 'Nonsense' ? '–' : predictedClass;
        predictionLetterElement.textContent = displayLetter;
        
        const conf = (pred.confidence * 100).toFixed(1);
        confidenceElement.textContent = `${conf}%`;
    } else if (data.current_prediction) {
        const displayLetter = data.current_prediction === 'Nonsense' ? '–' : data.current_prediction;
        predictionLetterElement.textContent = displayLetter;
        
        if (data.confidence) {
            const conf = (data.confidence * 100).toFixed(1);
            confidenceElement.textContent = `${conf}%`;
        }
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
    
    // Show general message
    if (data.message && !isRecording) {
        console.log('💬', data.message);
    }
}

// Update status badge
function updateStatus(status, text) {
    statusBadge.className = 'status-badge ' + status;
    statusText.textContent = text;
}

// Show success message
function showSuccess() {
    successMessage.style.display = 'flex';
    setTimeout(() => {
        successMessage.style.display = 'none';
    }, 2000);
}

// Show timeout message
function showTimeout() {
    timeoutMessage.style.display = 'flex';
    setTimeout(() => {
        timeoutMessage.style.display = 'none';
    }, 2000);
}

// Show hint message
function showHint(message) {
    hintText.textContent = message;
    hintMessage.style.display = 'flex';
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
