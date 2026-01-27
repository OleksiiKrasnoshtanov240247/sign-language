import torch
import numpy as np
from .model import ASLClassifier
from . import config

class SignLanguagePredictor:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else config.DEVICE
        
        # Load classes
        # Assuming classes.npy is in the same folder as config or relative to it
        # We might need absolute path if running from main.py
        # But config.LABEL_ENCODER_PATH is just 'classes.npy'.
        # Let's try to resolve it relative to config.py location
        base_path = config.BASE_DIR if hasattr(config, 'BASE_DIR') else list(config.__file__.split('\\')[:-1])
        if isinstance(base_path, list):
             # minimal fallback if Path not used in user's config
             import os
             base_dir = os.path.dirname(config.__file__)
             label_path = os.path.join(base_dir, config.LABEL_ENCODER_PATH)
        else:
             # If user config is simple and doesn't have BASE_DIR, construct it
             import os
             base_dir = os.path.dirname(config.__file__)
             label_path = os.path.join(base_dir, config.LABEL_ENCODER_PATH)

        try:
            self.classes = np.load(label_path, allow_pickle=True)
            print(f"Loaded classes: {self.classes}")
        except FileNotFoundError:
            print(f"Warning: classes.npy not found at {label_path}")
            self.classes = []

        # Initialize Model
        self.model = ASLClassifier(config.INPUT_SIZE, config.NUM_CLASSES).to(self.device)
        
        # Load Weights
        if model_path:
             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
             # Fallback to config path if not provided
             ckpt_path = os.path.join(base_dir, config.MODEL_SAVE_PATH)
             if os.path.exists(ckpt_path):
                 self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        
        self.model.eval()

    def predict(self, landmarks):
        """
        Predict from landmarks.
        Args:
            landmarks: np.array of shape (21, 3) or (63,)
        Returns:
            dict with 'predicted_class', 'confidence', 'all_probabilities'
        """
        # Flatten if needed
        if landmarks.ndim > 1:
            data = landmarks.flatten()
        else:
            data = landmarks.copy()
            
        # Add batch dim
        input_tensor = torch.tensor(data, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, idx = probs.max(dim=1)
            
            idx = idx.item()
            confidence = confidence.item()
            
            predicted_class = self.classes[idx] if idx < len(self.classes) else "Unknown"
            
            all_probs = {self.classes[i]: probs[0][i].item() for i in range(len(self.classes))}
            
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }