import torch
import torch.nn as nn


class SignLanguageMLP(nn.Module):
    """MLP for NGT fingerspelling classification from 63 landmark coordinates"""
    
    def __init__(self, input_dim=63, num_classes=24, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.7)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_features(self, x):
        """Extract features from penultimate layer for analysis"""
        for layer in self.network[:-1]:
            x = layer(x)
        return x
