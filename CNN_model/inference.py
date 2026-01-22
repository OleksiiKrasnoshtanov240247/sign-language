import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import numpy as np

from model import SignLanguageCNN, LightweightSignCNN


class SignLanguagePredictor:
    """
    Predictor class for Dutch Sign Language recognition.
    Load trained model and make predictions on new images.
    """

    def __init__(self, checkpoint_path, device=None):
        """
        Initialize predictor with trained model.

        Args:
            checkpoint_path: Path to saved model checkpoint
            device: torch device (defaults to cuda if available)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load class mapping
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Initialize model
        config = checkpoint.get('config', {})
        num_classes = len(self.class_to_idx)
        model_type = config.get('MODEL_TYPE', 'standard')
        dropout_rate = config.get('DROPOUT_RATE', 0.5)

        if model_type == "lightweight":
            self.model = LightweightSignCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        else:
            self.model = SignLanguageCNN(num_classes=num_classes, dropout_rate=dropout_rate)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get normalization parameters
        mean = config.get('MEAN', [0.485, 0.456, 0.406])
        std = config.get('STD', [0.229, 0.224, 0.225])
        img_size = config.get('IMG_SIZE', 224)

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Number of classes: {num_classes}")
        if 'val_acc' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['val_acc']:.4f}")

    def predict(self, image_path, top_k=3):
        """
        Make prediction on a single image.

        Args:
            image_path: Path to image file
            top_k: Return top k predictions

        Returns:
            Dictionary with predictions and probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.idx_to_class)))

        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'class': self.idx_to_class[idx.item()],
                'probability': prob.item()
            })

        return {
            'predictions': predictions,
            'top_prediction': predictions[0]['class'],
            'confidence': predictions[0]['probability']
        }

    def predict_batch(self, image_paths, batch_size=32):
        """
        Make predictions on multiple images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                image_tensor = self.transform(image)
                batch_images.append(image_tensor)

            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)

            for j, (path, probs) in enumerate(zip(batch_paths, probabilities)):
                pred_idx = torch.argmax(probs).item()
                results.append({
                    'image_path': str(path),
                    'predicted_class': self.idx_to_class[pred_idx],
                    'confidence': probs[pred_idx].item()
                })

        return results

    def predict_from_array(self, image_array):
        """
        Make prediction from numpy array (useful for webcam/video feed).

        Args:
            image_array: numpy array of shape (H, W, 3) in RGB format

        Returns:
            Dictionary with predictions and probabilities
        """
        # Convert numpy array to PIL Image
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array.astype('uint8'), 'RGB')
        else:
            image = image_array

        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        pred_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0][pred_idx].item()

        return {
            'predicted_class': self.idx_to_class[pred_idx],
            'confidence': confidence,
            'all_probabilities': {
                self.idx_to_class[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }


def main():
    """Example usage of the predictor"""
    import argparse

    parser = argparse.ArgumentParser(description='Dutch Sign Language Recognition - Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--top_k', type=int, default=3, help='Show top k predictions')

    args = parser.parse_args()

    # Initialize predictor
    predictor = SignLanguagePredictor(args.checkpoint)

    if args.image:
        # Predict single image
        result = predictor.predict(args.image, top_k=args.top_k)
        print(f"\nImage: {args.image}")
        print(f"Top prediction: {result['top_prediction']} ({result['confidence']:.4f})")
        print("\nAll predictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['class']}: {pred['probability']:.4f}")

    elif args.image_dir:
        # Predict directory of images
        image_paths = list(Path(args.image_dir).glob('*.jpg'))
        image_paths.extend(Path(args.image_dir).glob('*.png'))

        results = predictor.predict_batch(image_paths)

        print(f"\nProcessed {len(results)} images")
        for result in results:
            print(f"{Path(result['image_path']).name}: {result['predicted_class']} ({result['confidence']:.4f})")

    else:
        print("Please provide either --image or --image_dir argument")


if __name__ == "__main__":
    main()