import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from PIL import Image
from collections import Counter
import json


def analyze_dataset(data_dir, output_dir="dataset_analysis"):
    """
    Analyze dataset structure and statistics.

    Args:
        data_dir: Path to dataset directory (train or test)
        output_dir: Directory to save analysis results
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nAnalyzing dataset: {data_dir}")
    print("=" * 60)

    # Get all classes
    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes: {classes}")

    # Count images per class
    class_counts = {}
    image_sizes = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for class_name in classes:
        class_dir = data_path / class_name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_extensions]
        class_counts[class_name] = len(images)

        # Sample image sizes from first 10 images
        for img_path in images[:10]:
            try:
                img = Image.open(img_path)
                image_sizes.append(img.size)
            except:
                pass

    # Statistics
    total_images = sum(class_counts.values())
    avg_per_class = np.mean(list(class_counts.values()))
    std_per_class = np.std(list(class_counts.values()))
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())

    print(f"\nDataset Statistics:")
    print(f"  Total images: {total_images}")
    print(f"  Average per class: {avg_per_class:.1f} ± {std_per_class:.1f}")
    print(f"  Min count: {min_count}")
    print(f"  Max count: {max_count}")
    print(f"  Imbalance ratio: {max_count / min_count:.2f}:1")

    # Image size analysis
    if image_sizes:
        widths = [size[0] for size in image_sizes]
        heights = [size[1] for size in image_sizes]
        print(f"\nImage Size Statistics (sample of {len(image_sizes)} images):")
        print(f"  Width: {np.mean(widths):.0f} ± {np.std(widths):.0f} px")
        print(f"  Height: {np.mean(heights):.0f} ± {np.std(heights):.0f} px")

    # Class distribution
    print(f"\nClass Distribution:")
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_images) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {class_name:12} {count:5} ({percentage:5.1f}%) {bar}")

    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot
    classes_sorted = sorted(class_counts.keys())
    counts_sorted = [class_counts[c] for c in classes_sorted]

    axes[0].bar(range(len(classes_sorted)), counts_sorted, color='steelblue')
    axes[0].set_xticks(range(len(classes_sorted)))
    axes[0].set_xticklabels(classes_sorted, rotation=45, ha='right')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Class Distribution')
    axes[0].axhline(y=avg_per_class, color='r', linestyle='--', label=f'Average: {avg_per_class:.0f}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Pie chart
    axes[1].pie(counts_sorted, labels=classes_sorted, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Proportion')

    plt.tight_layout()
    plt.savefig(output_path / f"{data_path.name}_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nClass distribution plot saved to {output_path / f'{data_path.name}_distribution.png'}")

    # Save statistics to JSON
    stats = {
        'dataset_path': str(data_dir),
        'num_classes': len(classes),
        'total_images': total_images,
        'class_counts': class_counts,
        'statistics': {
            'average_per_class': float(avg_per_class),
            'std_per_class': float(std_per_class),
            'min_count': int(min_count),
            'max_count': int(max_count),
            'imbalance_ratio': float(max_count / min_count)
        }
    }

    with open(output_path / f"{data_path.name}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return class_counts


def visualize_samples(data_dir, output_dir="dataset_analysis", samples_per_class=5):
    """
    Visualize random samples from each class.

    Args:
        data_dir: Path to dataset directory
        output_dir: Directory to save visualizations
        samples_per_class: Number of samples to show per class
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=(samples_per_class * 3, num_classes * 3))

    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for i, class_name in enumerate(classes):
        class_dir = data_path / class_name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_extensions]

        # Random sample
        sample_images = np.random.choice(images,
                                         size=min(samples_per_class, len(images)),
                                         replace=False)

        for j, img_path in enumerate(sample_images):
            try:
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f"{class_name}", fontsize=12, fontweight='bold')
            except:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / f"{data_path.name}_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample visualization saved to {output_path / f'{data_path.name}_samples.png'}")


def compare_train_test(train_dir, test_dir, output_dir="dataset_analysis"):
    """
    Compare train and test set distributions.

    Args:
        train_dir: Path to training data
        test_dir: Path to test data
        output_dir: Directory to save comparison
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAIN vs TEST COMPARISON")
    print("=" * 60)

    train_counts = {}
    test_counts = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Get train counts
    train_path = Path(train_dir)
    for class_dir in train_path.iterdir():
        if class_dir.is_dir():
            count = len([f for f in class_dir.iterdir() if f.suffix.lower() in valid_extensions])
            train_counts[class_dir.name] = count

    # Get test counts
    test_path = Path(test_dir)
    for class_dir in test_path.iterdir():
        if class_dir.is_dir():
            count = len([f for f in class_dir.iterdir() if f.suffix.lower() in valid_extensions])
            test_counts[class_dir.name] = count

    # Find common classes
    train_classes = set(train_counts.keys())
    test_classes = set(test_counts.keys())
    common_classes = sorted(train_classes & test_classes)

    print(f"\nTrain classes: {len(train_classes)}")
    print(f"Test classes: {len(test_classes)}")
    print(f"Common classes: {len(common_classes)}")

    if train_classes != test_classes:
        print("\nWarning: Train and test sets have different classes!")
        if train_classes - test_classes:
            print(f"  Only in train: {train_classes - test_classes}")
        if test_classes - train_classes:
            print(f"  Only in test: {test_classes - train_classes}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(common_classes))
    width = 0.35

    train_vals = [train_counts.get(c, 0) for c in common_classes]
    test_vals = [test_counts.get(c, 0) for c in common_classes]

    ax.bar(x - width / 2, train_vals, width, label='Train', color='steelblue')
    ax.bar(x + width / 2, test_vals, width, label='Test', color='coral')

    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Images')
    ax.set_title('Train vs Test Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(common_classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "train_test_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {output_path / 'train_test_comparison.png'}")

    # Compute split ratios
    print(f"\nTrain/Test Split Ratios:")
    for class_name in common_classes:
        train_count = train_counts.get(class_name, 0)
        test_count = test_counts.get(class_name, 0)
        total = train_count + test_count
        train_ratio = (train_count / total * 100) if total > 0 else 0
        print(f"  {class_name:12} {train_ratio:5.1f}% / {100 - train_ratio:5.1f}%")


def main():
    """Main function for dataset analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze NGT dataset')
    parser.add_argument('--train_dir', type=str, help='Path to training directory')
    parser.add_argument('--test_dir', type=str, help='Path to test directory')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--visualize_samples', action='store_true',
                        help='Generate sample visualization')
    parser.add_argument('--samples_per_class', type=int, default=5,
                        help='Number of samples to visualize per class')

    args = parser.parse_args()

    if args.train_dir:
        analyze_dataset(args.train_dir, args.output_dir)
        if args.visualize_samples:
            visualize_samples(args.train_dir, args.output_dir, args.samples_per_class)

    if args.test_dir:
        analyze_dataset(args.test_dir, args.output_dir)
        if args.visualize_samples:
            visualize_samples(args.test_dir, args.output_dir, args.samples_per_class)

    if args.train_dir and args.test_dir:
        compare_train_test(args.train_dir, args.test_dir, args.output_dir)


if __name__ == "__main__":
    main()