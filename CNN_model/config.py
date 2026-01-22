import torch
from pathlib import Path

class Config:
    """Configuration for Dutch Sign Language CNN training"""
    BASE_DIR = Path(__file__).resolve().parent  # Gets the folder where this script lives
    TRAIN_DIR = BASE_DIR / "dataset" / "train"
    TEST_DIR = BASE_DIR / "dataset" / "test"

    # Model parameters
    NUM_CLASSES = 23  # Based on your dataset structure
    IMG_SIZE = 224  # Input image size (224x224)
    DROPOUT_RATE = 0.5

    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # Learning rate scheduler
    LR_SCHEDULER = "reduce_on_plateau"  # Options: "reduce_on_plateau", "step", "cosine"
    LR_PATIENCE = 5  # For ReduceLROnPlateau
    LR_FACTOR = 0.5  # For ReduceLROnPlateau
    STEP_SIZE = 10  # For StepLR
    STEP_GAMMA = 0.1  # For StepLR

    # Early stopping
    EARLY_STOPPING_PATIENCE = 10

    # Model selection
    MODEL_TYPE = "standard"  # Options: "standard", "lightweight"

    # Data augmentation parameters
    RANDOM_ROTATION = 15
    COLOR_JITTER = True
    HORIZONTAL_FLIP = 0.5  # Probability

    # Normalization (ImageNet stats by default)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Training settings
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpointing
    SAVE_DIR = "checkpoints"
    SAVE_BEST_ONLY = True

    # Logging
    LOG_INTERVAL = 10  # Log every N batches

    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision for faster training

    # Class names - matches your dataset structure (23 letters + nothing)
    CLASS_NAMES = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'U', 'V', 'W', 'X', 'Y'
    ]

    @classmethod
    def update_from_dict(cls, config_dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)