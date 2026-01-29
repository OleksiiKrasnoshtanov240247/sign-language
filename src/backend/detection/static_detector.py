"""
Static sign language detector using CNN for landmark classification.
Based on CNN_model/inference.py
"""
import torch
import numpy as np
from pathlib import Path
from src.backend.models.cnn_model import ASLClassifier
from src.backend.models import config


class StaticSignPredictor:
    """
    Predictor for static ASL signs (all letters except J and Z).
    Uses a trained CNN (MLP) to classify hand landmarks.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the static sign predictor.
        
        Args:
            model_path: Path to trained model weights (.pth file)
            device: torch.device to run inference on
        """
        self.device = device if device else config.DEVICE
        
        # Load class labels
        label_path = Path(model_path).parent / "classes.npy" if model_path else config.LABEL_ENCODER_PATH
        
        try:
            self.classes = np.load(label_path, allow_pickle=True)
            print(f"Loaded {len(self.classes)} classes: {self.classes}")
        except FileNotFoundError:
            print(f"Warning: classes.npy not found at {label_path}")
            self.classes = []

        # Initialize model
        self.model = ASLClassifier(config.INPUT_SIZE, config.NUM_CLASSES).to(self.device)
        
        # Load weights
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        else:
            # Try default path
            if config.MODEL_SAVE_PATH.exists():
                self.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=self.device))
                print(f"Loaded model weights from {config.MODEL_SAVE_PATH}")
            else:
                print(f"Warning: No model weights found at {config.MODEL_SAVE_PATH}")
        
        self.model.eval()

    def predict(self, landmarks):
        """
        Predict sign from hand landmarks.
        
        Args:
            landmarks: np.array of shape (21, 3) or (63,) - hand landmarks
            
        Returns:
            dict with:
                - predicted_class: str, predicted letter
                - confidence: float, confidence score for prediction
                - all_probabilities: dict, all class probabilities
        """
        # Flatten if needed
        if landmarks.ndim > 1:
            data = landmarks.flatten()
        else:
            data = landmarks.copy()
            
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(data, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, idx = probs.max(dim=1)
            
            idx = idx.item()
            confidence = confidence.item()
            
            predicted_class = self.classes[idx] if idx < len(self.classes) else "Unknown"
            
            # Get probabilities for all classes
            all_probs = {self.classes[i]: probs[0][i].item() for i in range(len(self.classes))}
            
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
