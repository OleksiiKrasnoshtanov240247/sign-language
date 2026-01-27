import numpy as np
import cv2
import torch
from src.backend.video.capture import HandCapture, normalize
from CNN_model.inference import SignLanguagePredictor

# J and Z need movement, rest are static
DYNAMIC_LETTERS = ['J', 'Z']

class SignDetector:
    def __init__(self, static_model_path=None, device=None):
        self.hand_capture = HandCapture()
        self.buffer = []
        
        # Initialize Static Model (CNN)
        self.static_predictor = None
        if static_model_path:
            try:
                self.static_predictor = SignLanguagePredictor(static_model_path, device=device)
                print(f"Static Model loaded from {static_model_path}")
            except Exception as e:
                print(f"Failed to load static model: {e}")
        
    def is_dynamic(self, letter):
        return letter.upper() in DYNAMIC_LETTERS

    def process_frame(self, frame, target_letter):
        """
        Process a frame and return separation of concerns:
        - landmarks: raw landmarks or None
        - status: text description of status
        - data: prepared data for model (if ready) or None
        """
        landmarks = self.hand_capture.extract_landmarks(frame)
        
        status = f"Letter: {target_letter} | Hand: NO"
        data = None
        prediction_info = None
        
        if landmarks is not None:
            norm = normalize(landmarks)
            status = f"Letter: {target_letter} | Hand: YES"
            
            if self.is_dynamic(target_letter):
                # --- Dynamic Letter Logic (J, Z) ---
                self.buffer.append(norm)
                if len(self.buffer) > 30:
                    self.buffer.pop(0)
                
                status += f" | Buffer: {len(self.buffer)}/30"
                
                if len(self.buffer) == 30:
                    data_array = np.array(self.buffer)
                    data = data_array 
                    status += f" | Ready: {data.shape}"
                    # TODO: LSTM prediction here
                    
            else:
                # --- Static Letter Logic (A, B, C...) ---
                self.buffer = [] # clear buffer
                
                if self.static_predictor:
                    # Predict using MLP (Landmarks)
                    # User's model expects raw landmarks flattened [x1, y1, z1, x2, ...]
                    try:
                        result = self.static_predictor.predict(landmarks)
                        predicted_class = result['predicted_class']
                        confidence = result['confidence']
                        
                        status += f" | Pred: {predicted_class} ({confidence:.2f})"
                        prediction_info = result
                        
                        # Verify against target
                        if predicted_class == target_letter:
                             status += " [MATCH]"
                    except Exception as e:
                        print(f"Prediction Error: {e}")
                        status += " | Err"
                else:
                     data = norm.flatten() 

        else:
            # Hand lost
            if self.is_dynamic(target_letter):
                pass
            
        return landmarks, status, data, prediction_info
    
    def clear_buffer(self):
        self.buffer = []
