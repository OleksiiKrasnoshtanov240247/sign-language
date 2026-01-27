"""
NGT Sign Language Detector
===========================
Static letters (A-Y) → MLP model
Dynamic letters (J, Z) → LSTM model
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

# LSTM model
from LSTM_model.inference import DynamicSignPredictor

# Dynamic letters
DYNAMIC_LETTERS = ['J', 'Z']


def is_dynamic(letter):
    return letter.upper() in DYNAMIC_LETTERS


def normalize(landmarks):
    """Center on wrist + scale by hand size."""
    lm = landmarks.copy()
    wrist = lm[0].copy()
    lm = lm - wrist
    scale = np.linalg.norm(lm[9])
    if scale > 0.001:
        lm = lm / scale
    return lm


class ASLClassifier(nn.Module):
    """Static sign classifier."""
    
    def __init__(self, input_size, num_classes):
        super(ASLClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class HandCapture:
    """Hand detection and visualization."""
    
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.last_landmarks = None
        self.last_hand_landmarks = None
    
    def detect(self, frame):
        """Returns (21, 3) landmarks or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        if not result.multi_hand_landmarks:
            self.last_landmarks = None
            self.last_hand_landmarks = None
            return None
        
        hand = result.multi_hand_landmarks[0]
        self.last_hand_landmarks = hand
        self.last_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        return self.last_landmarks
    
    def visualize_landmarks(self, frame):
        """Draw hand skeleton on frame."""
        if self.last_hand_landmarks is None:
            return
        
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            self.last_hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )
    
    def close(self):
        self.hands.close()


class SignDetector:
    """
    Main detector combining static and dynamic models.
    Compatible with cli.py interface.
    """
    
    def __init__(self, static_model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hand capture
        self.hand_capture = HandCapture()
        
        # Static model
        self.static_model = None
        self.static_classes = None
        if static_model_path:
            self._load_static_model(static_model_path)
        
        # Dynamic model (LSTM for J/Z)
        try:
            self.dynamic_predictor = DynamicSignPredictor()
        except Exception as e:
            print(f"⚠️ LSTM model not loaded: {e}")
            self.dynamic_predictor = None
        
        # Buffer for dynamic letters
        self.dynamic_buffer = []
        self.is_recording = False  # Only record when explicitly started
    
    def _load_static_model(self, model_path):
        """Load static letter model."""
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"⚠️ Static model not found: {model_path}")
            return
        
        # Load classes
        classes_path = model_path.parent / "classes.npy"
        if classes_path.exists():
            self.static_classes = np.load(classes_path, allow_pickle=True)
        else:
            self.static_classes = np.array([c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"])
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        input_size = 63
        num_classes = len(self.static_classes)
        self.static_model = ASLClassifier(input_size, num_classes)
        
        if 'model_state_dict' in checkpoint:
            self.static_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.static_model.load_state_dict(checkpoint)
        
        self.static_model.to(self.device)
        self.static_model.eval()
        print(f"✅ Static model loaded: {model_path}")
    
    def _predict_static(self, landmarks_flat):
        """Predict static letter."""
        if self.static_model is None:
            return None
        
        X = torch.tensor(landmarks_flat, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.static_model(X)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
        
        return {
            'predicted_letter': str(self.static_classes[pred_idx]),
            'confidence': confidence,
            'all_probabilities': {
                str(self.static_classes[i]): float(probs[i].item())
                for i in range(len(self.static_classes))
            }
        }
    
    def _predict_dynamic(self, landmarks_flat):
        """Add frame to dynamic buffer and predict if ready."""
        if self.dynamic_predictor is None:
            return None
        
        # Only record if explicitly started
        if not self.is_recording:
            return None
        
        self.dynamic_buffer.append(landmarks_flat)
        
        if len(self.dynamic_buffer) >= 30:
            # Make prediction
            self.dynamic_predictor.buffer.clear()
            for frame in self.dynamic_buffer[-30:]:
                self.dynamic_predictor.buffer.append(frame)
            
            result = self.dynamic_predictor.predict()
            
            # Stop recording and clear buffer
            self.is_recording = False
            self.dynamic_buffer = []
            
            if result:
                return {
                    'predicted_letter': result['letter'],
                    'confidence': result['confidence'],
                    'all_probabilities': result['probabilities']
                }
        
        return None
    
    def start_recording(self):
        """Start recording for dynamic letters (J/Z)."""
        self.dynamic_buffer = []
        self.is_recording = True
    
    def get_buffer_progress(self):
        """Get buffer fill progress (0-30)."""
        return len(self.dynamic_buffer)
    
    def process_frame(self, frame, target_letter):
        """
        Process a frame and return prediction.
        
        Returns:
            landmarks: (21, 3) or None
            status: str
            data: normalized landmarks
            prediction_info: dict or None
        """
        # Detect hand
        landmarks = self.hand_capture.detect(frame)
        
        if landmarks is None:
            status = f"Letter: {target_letter} | Hand: NO"
            return None, status, None, None
        
        # Normalize
        norm = normalize(landmarks)
        flat = norm.flatten().astype(np.float32)
        
        # Check if dynamic or static
        if is_dynamic(target_letter):
            # Dynamic letter (J, Z)
            prediction = self._predict_dynamic(flat)
            
            if prediction:
                pred_letter = prediction['predicted_letter']
                conf = prediction['confidence']
                match = "[MATCH]" if pred_letter == target_letter else "[NO MATCH]"
                status = f"Letter: {target_letter} | Hand: YES | Pred: {pred_letter} ({conf:.2f}) {match}"
            elif self.is_recording:
                buffer_len = len(self.dynamic_buffer)
                status = f"Letter: {target_letter} | Hand: YES | Recording: {buffer_len}/30"
                prediction = None
            else:
                status = f"Letter: {target_letter} | Hand: YES | Press SPACE to record"
                prediction = None
        else:
            # Static letter
            prediction = self._predict_static(flat)
            
            if prediction:
                pred_letter = prediction['predicted_letter']
                conf = prediction['confidence']
                match = "[MATCH]" if pred_letter == target_letter else "[NO MATCH]"
                status = f"Letter: {target_letter} | Hand: YES | Pred: {pred_letter} ({conf:.2f}) {match}"
            else:
                status = f"Letter: {target_letter} | Hand: YES | No model"
        
        return landmarks, status, flat, prediction
    
    def clear_buffer(self):
        """Clear dynamic buffer."""
        self.dynamic_buffer = []
        self.is_recording = False
        if self.dynamic_predictor:
            self.dynamic_predictor.clear_buffer()
    
    def close(self):
        """Cleanup."""
        self.hand_capture.close()


# Test
if __name__ == "__main__":
    detector = SignDetector()
    cap = cv2.VideoCapture(0)
    
    target = 'A'
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        landmarks, status, data, pred = detector.process_frame(frame, target)
        
        detector.hand_capture.visualize_landmarks(frame)
        
        color = (0, 255, 0) if "Hand: YES" in status else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('a') <= key <= ord('z'):
            target = chr(key).upper()
            detector.clear_buffer()
    
    detector.close()
    cap.release()
    cv2.destroyAllWindows()