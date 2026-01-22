import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path


class SignLanguageDataset(Dataset):
    """
    Dataset class for Dutch Sign Language images.
    Expects folder structure:
    train/
        A/
            image1.jpg
            image2.jpg
        B/
            image1.jpg
        ...
    """

    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Get all image paths and labels
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])

        # Create class to index mapping
        if class_to_idx is None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])

        # Collect all image paths
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")

        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self):
        """Return distribution of classes in the dataset"""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts


def get_transforms(img_size, augment=True, mean=None, std=None):
    """
    Get data transforms for training or validation.

    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation
        mean: Normalization mean
        std: Normalization std
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    if augment:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transform


def get_data_loaders(config):
    """
    Create train and test data loaders.

    Args:
        config: Configuration object with data paths and parameters

    Returns:
        train_loader, test_loader, class_to_idx
    """
    # Training transforms with augmentation
    train_transform = get_transforms(
        img_size=config.IMG_SIZE,
        augment=True,
        mean=config.MEAN,
        std=config.STD
    )

    # Validation/test transforms without augmentation
    test_transform = get_transforms(
        img_size=config.IMG_SIZE,
        augment=False,
        mean=config.MEAN,
        std=config.STD
    )

    # Create datasets
    train_dataset = SignLanguageDataset(
        root_dir=config.TRAIN_DIR,
        transform=train_transform
    )

    test_dataset = SignLanguageDataset(
        root_dir=config.TEST_DIR,
        transform=test_transform,
        class_to_idx=train_dataset.class_to_idx  # Use same mapping as train
    )

    # Print class distributions
    print("\nTraining set distribution:")
    train_dist = train_dataset.get_class_distribution()
    for class_name, count in sorted(train_dist.items()):
        print(f"  {class_name}: {count}")

    print("\nTest set distribution:")
    test_dist = test_dataset.get_class_distribution()
    for class_name, count in sorted(test_dist.items()):
        print(f"  {class_name}: {count}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, test_loader, train_dataset.class_to_idx


def compute_class_weights(data_loader, num_classes):
    """
    Compute class weights for imbalanced datasets.

    Args:
        data_loader: DataLoader object
        num_classes: Number of classes

    Returns:
        torch.Tensor: Class weights
    """
    class_counts = torch.zeros(num_classes)

    for _, labels in data_loader:
        for label in labels:
            class_counts[label] += 1

    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes

    return class_weights