"""
Neural Network Architecture for Sign Language Classification
============================================================
ResidualMLP: MLP with skip connections for better gradient flow
Input: 63 dimensions (21 landmarks × 3 coordinates)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection for deeper networks."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        return F.gelu(x + self.block(x))


class ResidualMLP(nn.Module):
    """
    MLP with Residual connections.
    Better gradient flow for deeper architectures.
    
    Args:
        input_dim: Number of input features (default: 63 for hand landmarks)
        num_classes: Number of output classes (default: 25 for NGT alphabet + Nonsense)
        hidden_dim: Hidden layer dimension
        num_blocks: Number of residual blocks
        dropout: Dropout probability
    """
    def __init__(self, input_dim=63, num_classes=25, hidden_dim=256, num_blocks=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.classifier(x)


def load_model(model_path, device):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to .pth checkpoint file
        device: torch device (cuda/cpu)
    
    Returns:
        model: Loaded model in eval mode
        model_name: Name of the model architecture
        test_acc: Test accuracy from training
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    num_classes = checkpoint['num_classes']
    input_dim = checkpoint.get('input_dim', 63)
    
    model = ResidualMLP(input_dim, num_classes, hidden_dim=256, num_blocks=4, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('model_name', 'ResidualMLP'), checkpoint.get('test_acc', 0)