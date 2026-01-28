import cv2
import torch
import numpy as np
from collections import Counter, deque
import mediapipe as mp
from model import SignLanguageMLP
import pickle


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

    Input: (21, 3) numpy array
    Output: (63,) flattened array
    """
    lm = landmarks.copy()
    lm -= lm[0]

    scale = np.linalg.norm(lm[9])
    if scale > 0.001:
        lm /= scale

    return lm.flatten()


def calculate_entropy(probabilities):
    """Calculate Shannon entropy of probability distribution"""
    return -np.sum(probabilities * np.log(probabilities + 1e-10))


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

    ENTROPY_THRESHOLD = 1.5
    CONFIDENCE_THRESHOLD = 0.6
    SMOOTHING_WINDOW = 5

    MIN_FRAMES_BEFORE_PREDICT = 45
    RESET_FRAMES_NO_HAND = 15

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

    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Classes ({len(label_encoder.classes_)}): {label_encoder.classes_}")

    print("Initializing MediaPipe...")
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    smoother = PredictionSmoother(window_size=SMOOTHING_WINDOW)

    frames_with_hand = 0
    frames_without_hand = 0
    in_recognition_phase = False

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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            frames_with_hand += 1
            frames_without_hand = 0

            if frames_with_hand >= MIN_FRAMES_BEFORE_PREDICT:
                in_recognition_phase = True

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if not in_recognition_phase:
                progress_pct = int(100 * frames_with_hand / MIN_FRAMES_BEFORE_PREDICT)
                status_lines = [
                    "Form your sign...",
                    f"Preparing: {progress_pct}%",
                    f"Hold steady ({frames_with_hand}/{MIN_FRAMES_BEFORE_PREDICT})"
                ]
                draw_status_box(frame, "PREPARING", (200, 200, 200), status_lines)
            else:
                landmarks = hand_landmarks.landmark
                coordinates = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

                landmarks_normalized = normalize_landmarks(coordinates)

                with torch.no_grad():
                    x_tensor = torch.FloatTensor(landmarks_normalized).unsqueeze(0).to(device)
                    outputs = model(x_tensor)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                predicted_class = probabilities.argmax()
                confidence = probabilities[predicted_class]
                entropy = calculate_entropy(probabilities)

                if frame_count % 30 == 0:
                    top3_idx = np.argsort(probabilities)[-3:][::-1]
                    top3_letters = label_encoder.inverse_transform(top3_idx)
                    top3_probs = probabilities[top3_idx]
                    print(f"[Frame {frame_count}] {top3_letters[0]}({top3_probs[0]:.2f}), "
                          f"{top3_letters[1]}({top3_probs[1]:.2f}), {top3_letters[2]}({top3_probs[2]:.2f}) | "
                          f"Entropy: {entropy:.2f}")

                if entropy < ENTROPY_THRESHOLD and confidence > CONFIDENCE_THRESHOLD:
                    smoother.add(predicted_class)
                    smoothed = smoother.get_smoothed()

                    if smoothed is not None:
                        letter = label_encoder.inverse_transform([smoothed])[0]

                        if letter == "Nonsense":
                            status_lines = [
                                "Not a valid sign",
                                f"Confidence: {confidence:.2f}",
                                "Try a clearer gesture"
                            ]
                            draw_status_box(frame, "NONSENSE", (255, 165, 0), status_lines)
                        else:
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
            frames_without_hand += 1
            frames_with_hand = 0

            if frames_without_hand >= RESET_FRAMES_NO_HAND:
                in_recognition_phase = False
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
            frames_with_hand = 0
            in_recognition_phase = False
            print("Reset: smoothing buffer + recognition phase")
        elif key == ord('s') and results.multi_hand_landmarks:
            import time
            timestamp = int(time.time())
            cv2.imwrite(f"debug_frame_{timestamp}.png", frame)
            np.save(f"debug_landmarks_{timestamp}.npy", landmarks_normalized)
            print(f"\nSaved debug_frame_{timestamp}.png and debug_landmarks_{timestamp}.npy")
            print(f"Predicted: {label_encoder.inverse_transform([predicted_class])[0]} (conf: {confidence:.2f})")

        frame_count += 1

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nInference stopped.")


if __name__ == "__main__":
    main()