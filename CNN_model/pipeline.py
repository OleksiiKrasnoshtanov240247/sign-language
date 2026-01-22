import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import time

from model import SignLanguageCNN, LightweightSignCNN
from data_loader import get_data_loaders, compute_class_weights
from config import Config


class Trainer:
    """Training manager for sign language CNN"""

    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        # Create save directory
        self.save_dir = Path(config.SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize model
        if config.MODEL_TYPE == "lightweight":
            self.model = LightweightSignCNN(
                num_classes=config.NUM_CLASSES,
                dropout_rate=config.DROPOUT_RATE
            )
        else:
            self.model = SignLanguageCNN(
                num_classes=config.NUM_CLASSES,
                dropout_rate=config.DROPOUT_RATE
            )

        self.model = self.model.to(self.device)

        # Get data loaders
        self.train_loader, self.test_loader, self.class_to_idx = get_data_loaders(config)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Save class mapping
        with open(self.save_dir / "class_mapping.json", "w") as f:
            json.dump(self.class_to_idx, f, indent=2)

        # Compute class weights for imbalanced datasets
        class_weights = compute_class_weights(self.train_loader, config.NUM_CLASSES)
        class_weights = class_weights.to(self.device)

        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        if config.LR_SCHEDULER == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config.LR_FACTOR,
                patience=config.LR_PATIENCE
            )
        elif config.LR_SCHEDULER == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.STEP_SIZE,
                gamma=config.STEP_GAMMA
            )
        elif config.LR_SCHEDULER == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS
            )
        else:
            self.scheduler = None

        # Mixed precision training
        self.scaler = GradScaler() if config.USE_AMP else None

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{running_loss / (batch_idx + 1):.4f}"
                })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.test_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        # Compute detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config.__dict__,
            'class_to_idx': self.class_to_idx
        }

        if is_best:
            path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, path)
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")

        # Always save last checkpoint
        last_path = self.save_dir / "last_checkpoint.pth"
        torch.save(checkpoint, last_path)

    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.config.CLASS_NAMES,
            yticklabels=self.config.CLASS_NAMES
        )
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'confusion_matrix_epoch_{epoch}.png', dpi=150)
        plt.close()

    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=150)
        plt.close()

    def train(self):
        """Main training loop"""
        print(f"\nTraining on device: {self.device}")
        print(f"Model type: {self.config.MODEL_TYPE}")
        print(f"Number of classes: {self.config.NUM_CLASSES}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Number of epochs: {self.config.NUM_EPOCHS}\n")

        for epoch in range(self.config.NUM_EPOCHS):
            start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print("-" * 60)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, precision, recall, f1, val_preds, val_labels = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Print metrics
            epoch_time = time.time() - start_time
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            print(f"Epoch time: {epoch_time:.2f}s")

            # Learning rate scheduling
            if self.scheduler:
                if self.config.LR_SCHEDULER == "reduce_on_plateau":
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.save_checkpoint(epoch + 1, val_acc, is_best=True)

                # Plot confusion matrix for best model
                self.plot_confusion_matrix(val_labels, val_preds, epoch + 1)
            else:
                self.patience_counter += 1

            # Save last checkpoint
            if not self.config.SAVE_BEST_ONLY:
                self.save_checkpoint(epoch + 1, val_acc, is_best=False)

            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
                break

        # Plot final training history
        self.plot_training_history()

        # Save training history
        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        print(f"Models saved to: {self.save_dir}")
        print(f"{'=' * 60}")


def main():
    """Main function to run training"""
    # Load configuration
    config = Config()

    # Verify paths exist
    if not Path(config.TRAIN_DIR).exists():
        raise ValueError(f"Training directory does not exist: {config.TRAIN_DIR}")
    if not Path(config.TEST_DIR).exists():
        raise ValueError(f"Test directory does not exist: {config.TEST_DIR}")

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()