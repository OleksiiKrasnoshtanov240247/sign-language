import cv2
import torch
import numpy as np
from collections import Counter, deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model import SignLanguageMLP


class PredictionSmoother:
    def __init__(self, window_size=5, min_agreement=0.6):
        self.window_size = window_size
        self.min_agreement = min_agreement
        self.buffer = deque(maxlen=window_size)

    def add(self, prediction):
        self.buffer.append(prediction)

    def get_smoothed(self):
        if len(self.buffer) < self.window_size:
            return None

        counts = Counter(self.buffer)
        most_common, count = counts.most_common(1)[0]

        if count >= self.window_size * self.min_agreement:
            return most_common
        return None

    def reset(self):
        self.buffer.clear()


def normalize_landmarks(landmarks):
    """
    Normalize landmarks to match training preprocessing:
    1. Center on wrist (landmark 0)
    2. Scale by distance wrist -> middle finger base (landmark 9)
    """
    points = landmarks.reshape(21, 3)

    wrist = points[0].copy()
    points = points - wrist

    scale_factor = np.linalg.norm(points[9])
    if scale_factor > 0.001:
        points = points / scale_factor

    return points.flatten()


def calculate_entropy(probabilities):
    """Calculate Shannon entropy of probability distribution"""
    return -np.sum(probabilities * np.log(probabilities + 1e-10))


def draw_landmarks(frame, hand_landmarks, mp_hands, flipped=True):
    """Draw hand landmarks and connections on frame"""
    h, w, _ = frame.shape

    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]

        start_point = hand_landmarks[start_idx]
        end_point = hand_landmarks[end_idx]

        # If frame is flipped, mirror the x coordinates for drawing
        if flipped:
            start_px = (int((1 - start_point.x) * w), int(start_point.y * h))
            end_px = (int((1 - end_point.x) * w), int(end_point.y * h))
        else:
            start_px = (int(start_point.x * w), int(start_point.y * h))
            end_px = (int(end_point.x * w), int(end_point.y * h))

        cv2.line(frame, start_px, end_px, (0, 255, 0), 2)

    for landmark in hand_landmarks:
        if flipped:
            cx, cy = int((1 - landmark.x) * w), int(landmark.y * h)
        else:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)


def draw_status_box(frame, status, color, text_lines):
    """Draw status box with prediction info"""
    h, w, _ = frame.shape
    box_h = 120

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, box_h), color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.rectangle(frame, (10, 10), (w - 10, box_h), color, 2)

    y_offset = 40
    for line in text_lines:
        cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30


def main():
    MODEL_PATH = "best_model.pth"
    LABEL_ENCODER_PATH = "label_encoder.pkl"
    MEDIAPIPE_MODEL = "hand_landmarker.task"

    ENTROPY_THRESHOLD = 1.5
    CONFIDENCE_THRESHOLD = 0.6
    SMOOTHING_WINDOW = 5

    print("=" * 60)
    print("NGT SIGN LANGUAGE - WEBCAM INFERENCE")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = SignLanguageMLP(
        input_dim=checkpoint['input_dim'],
        num_classes=checkpoint['num_classes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    import pickle
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Classes: {label_encoder.classes_}")

    print("Initializing MediaPipe...")
    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    mp_hands = mp.solutions.hands
    smoother = PredictionSmoother(window_size=SMOOTHING_WINDOW)

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  Q - Quit")
    print("  R - Reset smoothing buffer")
    print("  S - Save current frame and landmarks (for debugging)")
    print("=" * 60)
    print("\nStarting inference...\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # CRITICAL: Don't flip frame before MediaPipe detection
        # Training data was extracted from unflipped images
        # Flip AFTER detection to match user's mirror view expectation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect(mp_image)

        # Now flip for display (landmarks will be mirrored for drawing)
        frame = cv2.flip(frame, 1)

        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand_landmarks = results.hand_landmarks[0]

            draw_landmarks(frame, hand_landmarks, mp_hands, flipped=True)

            coords = []
            for lm in hand_landmarks:
                coords.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(coords, dtype=np.float32)

            # DEBUG: Print raw landmarks
            if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                print(f"\n[DEBUG Frame {frame_count}]")
                print(f"  Raw wrist: [{landmarks[0]:.3f}, {landmarks[1]:.3f}, {landmarks[2]:.3f}]")
                print(f"  Raw point9: [{landmarks[27]:.3f}, {landmarks[28]:.3f}, {landmarks[29]:.3f}]")

            landmarks_normalized = normalize_landmarks(landmarks)

            # DEBUG: Print normalized landmarks
            if frame_count % 30 == 0:
                print(
                    f"  Normalized wrist: [{landmarks_normalized[0]:.3f}, {landmarks_normalized[1]:.3f}, {landmarks_normalized[2]:.3f}]")
                print(f"  Normalized point9 dist: {np.linalg.norm(landmarks_normalized[27:30]):.3f}")
                print(f"  Expected: wrist=[0,0,0], point9_dist=1.0")

            with torch.no_grad():
                x_tensor = torch.FloatTensor(landmarks_normalized).unsqueeze(0).to(device)
                outputs = model(x_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            predicted_class = probabilities.argmax()
            confidence = probabilities[predicted_class]
            entropy = calculate_entropy(probabilities)

            # DEBUG: Print predictions every 30 frames
            if frame_count % 30 == 0:
                top3_idx = np.argsort(probabilities)[-3:][::-1]
                top3_letters = label_encoder.inverse_transform(top3_idx)
                top3_probs = probabilities[top3_idx]
                print(
                    f"  Predictions: {top3_letters[0]}({top3_probs[0]:.2f}), {top3_letters[1]}({top3_probs[1]:.2f}), {top3_letters[2]}({top3_probs[2]:.2f})")
                print(f"  Entropy: {entropy:.2f}")

            if entropy < ENTROPY_THRESHOLD and confidence > CONFIDENCE_THRESHOLD:
                smoother.add(predicted_class)
                smoothed = smoother.get_smoothed()

                if smoothed is not None:
                    letter = label_encoder.inverse_transform([smoothed])[0]
                    status_lines = [
                        f"Letter: {letter}",
                        f"Confidence: {confidence:.2f}",
                        f"Entropy: {entropy:.2f}"
                    ]
                    draw_status_box(frame, "VALID", (0, 255, 0), status_lines)
                else:
                    status_lines = [
                        "Stabilizing...",
                        f"Confidence: {confidence:.2f}",
                        f"Entropy: {entropy:.2f}"
                    ]
                    draw_status_box(frame, "WAIT", (0, 255, 255), status_lines)
            else:
                smoother.reset()
                status_lines = [
                    "Uncertain gesture",
                    f"Entropy: {entropy:.2f}",
                    f"Confidence: {confidence:.2f}"
                ]
                draw_status_box(frame, "UNCERTAIN", (0, 165, 255), status_lines)
        else:
            smoother.reset()
            status_lines = [
                "No hand detected",
                "",
                "Show your hand to camera"
            ]
            draw_status_box(frame, "NO HAND", (128, 128, 128), status_lines)

        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('NGT Sign Language Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            smoother.reset()
            print("Smoothing buffer reset")
        elif key == ord('s') and results.hand_landmarks and len(results.hand_landmarks) > 0:
            # Save frame and landmarks for debugging
            import time
            timestamp = int(time.time())
            cv2.imwrite(f"debug_frame_{timestamp}.png", frame)
            np.save(f"debug_landmarks_{timestamp}.npy", landmarks_normalized)
            print(f"\nSaved debug_frame_{timestamp}.png and debug_landmarks_{timestamp}.npy")
            print(f"Predicted: {label_encoder.inverse_transform([predicted_class])[0]} (conf: {confidence:.2f})")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nInference stopped.")


if __name__ == "__main__":
    main()