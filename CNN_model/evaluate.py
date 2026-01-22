import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm

from model import SignLanguageCNN, LightweightSignCNN
from data_loader import get_data_loaders
from config import Config


class ModelEvaluator:
    """Comprehensive evaluation of trained model"""

    def __init__(self, checkpoint_path, config):
        self.config = config
        self.device = config.DEVICE

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

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

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load data
        _, self.test_loader, self.class_to_idx = get_data_loaders(config)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        print(f"Model loaded from {checkpoint_path}")
        print(f"Validation accuracy from training: {checkpoint.get('val_acc', 'N/A')}")

    def evaluate(self):
        """Run complete evaluation"""
        print("\nRunning evaluation...")

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def compute_metrics(self, y_true, y_pred):
        """Compute detailed metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Overall metrics
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        metrics = {
            'overall': {
                'accuracy': accuracy,
                'precision': precision_avg,
                'recall': recall_avg,
                'f1_score': f1_avg
            },
            'per_class': {}
        }

        for idx, class_name in self.idx_to_class.items():
            metrics['per_class'][class_name] = {
                'precision': precision[idx],
                'recall': recall[idx],
                'f1_score': f1[idx],
                'support': int(support[idx])
            }

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        """Plot detailed confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        # Raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.config.CLASS_NAMES,
            yticklabels=self.config.CLASS_NAMES,
            ax=axes[0],
            cbar_kws={'label': 'Count'}
        )
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)

        # Normalized
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.config.CLASS_NAMES,
            yticklabels=self.config.CLASS_NAMES,
            ax=axes[1],
            cbar_kws={'label': 'Proportion'}
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to {save_path}")

    def plot_per_class_metrics(self, metrics, save_path):
        """Plot per-class performance metrics"""
        classes = list(metrics['per_class'].keys())
        precision = [metrics['per_class'][c]['precision'] for c in classes]
        recall = [metrics['per_class'][c]['recall'] for c in classes]
        f1 = [metrics['per_class'][c]['f1_score'] for c in classes]
        support = [metrics['per_class'][c]['support'] for c in classes]

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Precision
        axes[0, 0].bar(range(len(classes)), precision, color='skyblue')
        axes[0, 0].set_xticks(range(len(classes)))
        axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision per Class')
        axes[0, 0].axhline(y=metrics['overall']['precision'], color='r', linestyle='--', label='Average')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Recall
        axes[0, 1].bar(range(len(classes)), recall, color='lightcoral')
        axes[0, 1].set_xticks(range(len(classes)))
        axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall per Class')
        axes[0, 1].axhline(y=metrics['overall']['recall'], color='r', linestyle='--', label='Average')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)

        # F1-Score
        axes[1, 0].bar(range(len(classes)), f1, color='lightgreen')
        axes[1, 0].set_xticks(range(len(classes)))
        axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('F1-Score per Class')
        axes[1, 0].axhline(y=metrics['overall']['f1_score'], color='r', linestyle='--', label='Average')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Support
        axes[1, 1].bar(range(len(classes)), support, color='plum')
        axes[1, 1].set_xticks(range(len(classes)))
        axes[1, 1].set_xticklabels(classes, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_title('Support per Class')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Per-class metrics plot saved to {save_path}")

    def analyze_misclassifications(self, y_true, y_pred, top_n=10):
        """Analyze most common misclassifications"""
        misclassified = y_true != y_pred

        # Count misclassification pairs
        misclass_pairs = {}
        for true_label, pred_label in zip(y_true[misclassified], y_pred[misclassified]):
            pair = (self.idx_to_class[true_label], self.idx_to_class[pred_label])
            misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1

        # Sort by frequency
        sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_n} Misclassification Pairs:")
        print("-" * 60)
        for i, ((true_class, pred_class), count) in enumerate(sorted_pairs[:top_n], 1):
            print(f"{i}. True: {true_class:8} -> Predicted: {pred_class:8} (Count: {count})")

        return sorted_pairs[:top_n]

    def compute_top_k_accuracy(self, y_true, y_probs, k_values=[1, 3, 5]):
        """Compute top-k accuracy"""
        top_k_accuracies = {}

        for k in k_values:
            top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
            correct = 0
            for i, true_label in enumerate(y_true):
                if true_label in top_k_preds[i]:
                    correct += 1
            top_k_accuracies[k] = correct / len(y_true)

        return top_k_accuracies

    def run_full_evaluation(self, output_dir="evaluation_results"):
        """Run complete evaluation pipeline"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Get predictions
        y_pred, y_true, y_probs = self.evaluate()

        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred)

        # Print overall metrics
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['overall']['accuracy']:.4f}")
        print(f"  Precision: {metrics['overall']['precision']:.4f}")
        print(f"  Recall:    {metrics['overall']['recall']:.4f}")
        print(f"  F1-Score:  {metrics['overall']['f1_score']:.4f}")

        # Top-k accuracy
        top_k_acc = self.compute_top_k_accuracy(y_true, y_probs)
        print(f"\nTop-K Accuracy:")
        for k, acc in top_k_acc.items():
            print(f"  Top-{k}: {acc:.4f}")

        # Per-class metrics
        print("\nPer-Class Metrics:")
        print("-" * 80)
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        for class_name in sorted(metrics['per_class'].keys()):
            m = metrics['per_class'][class_name]
            print(f"{class_name:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                  f"{m['f1_score']:<12.4f} {m['support']:<10}")

        # Analyze misclassifications
        misclass_pairs = self.analyze_misclassifications(y_true, y_pred)

        # Plot confusion matrix
        self.plot_confusion_matrix(
            y_true, y_pred,
            output_dir / "confusion_matrix.png"
        )

        # Plot per-class metrics
        self.plot_per_class_metrics(
            metrics,
            output_dir / "per_class_metrics.png"
        )

        # Save metrics to JSON
        metrics_output = {
            'overall_metrics': metrics['overall'],
            'per_class_metrics': metrics['per_class'],
            'top_k_accuracy': top_k_acc,
            'top_misclassifications': [
                {'true': pair[0][0], 'predicted': pair[0][1], 'count': pair[1]}
                for pair in misclass_pairs
            ]
        }

        with open(output_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics_output, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Evaluation complete! Results saved to {output_dir}")
        print(f"{'=' * 60}\n")

        return metrics


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Load config
    config = Config()

    # Run evaluation
    evaluator = ModelEvaluator(args.checkpoint, config)
    evaluator.run_full_evaluation(args.output_dir)


if __name__ == "__main__":
    main()