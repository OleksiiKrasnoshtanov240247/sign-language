import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from pathlib import Path
from tqdm import tqdm
from model import SignLanguageMLP


class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_split_data(csv_path, test_size=0.15, val_size=0.15):
    """Load CSV and create stratified train/val/test splits"""
    df = pd.read_csv(csv_path)
    
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df['label'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
    )
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_targets


def print_per_class_accuracy(y_true, y_pred, label_encoder):
    """Print accuracy for each letter"""
    from collections import defaultdict
    
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    
    for true, pred in zip(y_true, y_pred):
        total_per_class[true] += 1
        if true == pred:
            correct_per_class[true] += 1
    
    print("\nPer-class accuracy:")
    print("-" * 40)
    for class_idx in sorted(total_per_class.keys()):
        letter = label_encoder.inverse_transform([class_idx])[0]
        acc = 100. * correct_per_class[class_idx] / total_per_class[class_idx]
        print(f"  {letter}: {acc:.2f}% ({correct_per_class[class_idx]}/{total_per_class[class_idx]})")


def main():
    CSV_PATH = "ngt_final.csv"
    OUTPUT_DIR = Path("trained_models")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    PATIENCE = 15
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("NGT SIGN LANGUAGE CLASSIFIER - TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_and_split_data(CSV_PATH)
    
    train_dataset = LandmarkDataset(X_train, y_train)
    val_dataset = LandmarkDataset(X_val, y_val)
    test_dataset = LandmarkDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = SignLanguageMLP(input_dim=63, num_classes=len(label_encoder.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'input_dim': 63,
                'num_classes': len(label_encoder.classes_)
            }, OUTPUT_DIR / "best_model.pth")
            print(f"  -> New best model saved (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered (patience={PATIENCE})")
            break
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    checkpoint = torch.load(OUTPUT_DIR / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_targets = validate(model, test_loader, criterion, DEVICE)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    print_per_class_accuracy(test_targets, test_preds, label_encoder)
    
    cm = confusion_matrix(test_targets, test_preds)
    np.save(OUTPUT_DIR / "confusion_matrix.npy", cm)
    print(f"\nConfusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.npy'}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: {OUTPUT_DIR / 'best_model.pth'}")
    print(f"Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
