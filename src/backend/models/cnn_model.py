"""
CNN (MLP) model architecture for static sign language classification.
Migrated from CNN_model/model.py
"""
import torch.nn as nn


class ASLClassifier(nn.Module):
    """
    Multi-layer perceptron for classifying static ASL signs from hand landmarks.
    
    Architecture:
        - Input: 63 features (21 landmarks * 3 coordinates)
        - Hidden layers: 128 -> 64 -> 32
        - Output: NUM_CLASSES (24 for A-Z excluding J, Z)
    """
    
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
