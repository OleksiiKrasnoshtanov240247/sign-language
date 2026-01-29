"""
Real-time Inference for Sign Language Recognition
=================================================
Uses webcam + MediaPipe for hand detection, then classifies with trained model.
Features: prediction smoothing, entropy filtering, nonsense rejection.
"""

import cv2
import torch
import numpy as np
from collections import Counter, deque
import mediapipe as mp
import pickle
from pathlib import Path

from models import load_model
from dataset_creation import normalize_landmarks

# ============ CONFIG ============
DATA_DIR = Path(".")
MODEL_PATH = DATA_DIR / "best_model.pth"
ENCODER_PATH = DATA_DIR / "label_encoder.pkl"

CONFIDENCE_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 1.8
SMOOTHING_WINDOW = 15          # Frames to consider for voting
MIN_AGREEMENT = 0.5            # 50% to confirm first letter
SWITCH_THRESHOLD = 0.6         # 60% to switch to new letter  
MIN_FRAMES_BEFORE_SWITCH = 10  # Minimum frames before allowing switch
MIN_HAND_FRAMES = 10


class PredictionSmoother:
    """
    Temporal smoothing for stable predictions.
    Uses "sticky" behavior - current letter stays until a new one dominates.
    """
    
    def __init__(self, window_size=15, min_agreement=0.5, switch_threshold=0.6):
        """
        Args:
            window_size: Number of frames to consider (default: 15)
            min_agreement: Min ratio to confirm first prediction (default: 50%)
            switch_threshold: Min ratio to switch to new letter (default: 60%)
        """
        self.window_size = window_size
        self.min_agreement = min_agreement
        self.switch_threshold = switch_threshold
        self.buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)
        self.current_letter = None  # Sticky letter
        self.frames_since_switch = 0
        self.min_frames_before_switch = 10  # Minimum frames before allowing switch
    
    def add(self, prediction, confidence):
        self.buffer.append(prediction)
        self.confidence_buffer.append(confidence)
        self.frames_since_switch += 1
    
    def get_smoothed(self):
        if len(self.buffer) < self.window_size // 2:  # Need at least half buffer
            return None, 0
        
        counts = Counter(self.buffer)
        most_common, count = counts.most_common(1)[0]
        ratio = count / len(self.buffer)
        
        # Calculate average confidence for most common
        avg_conf = np.mean([c for p, c in zip(self.buffer, self.confidence_buffer) if p == most_common])
        
        # First prediction - need min_agreement
        if self.current_letter is None:
            if ratio >= self.min_agreement:
                self.current_letter = most_common
                self.frames_since_switch = 0
                return most_common, avg_conf
            return None, 0
        
        # Same letter - keep it
        if most_common == self.current_letter:
            self.frames_since_switch = 0
            return self.current_letter, avg_conf
        
        # Different letter - need higher threshold AND enough frames
        if ratio >= self.switch_threshold and self.frames_since_switch >= self.min_frames_before_switch:
            self.current_letter = most_common
            self.frames_since_switch = 0
            return most_common, avg_conf
        
        # Keep current letter (sticky)
        current_conf = np.mean([c for p, c in zip(self.buffer, self.confidence_buffer) if p == self.current_letter]) if self.current_letter in counts else avg_conf * 0.5
        return self.current_letter, current_conf
    
    def reset(self):
        self.buffer.clear()
        self.confidence_buffer.clear()
        self.current_letter = None
        self.frames_since_switch = 0


def calculate_entropy(probs):
    """Calculate Shannon entropy of probability distribution."""
    return -np.sum(probs * np.log(probs + 1e-10))


def draw_ui(frame, letter, confidence, entropy, top3, status, model_name, fps):
    """Draw user interface overlay on frame."""
    h, w = frame.shape[:2]
    
    colors = {
        'valid': (0, 255, 0),
        'nonsense': (0, 165, 255),
        'uncertain': (0, 200, 255),
        'stabilizing': (255, 255, 0),
        'no_hand': (128, 128, 128)
    }
    color = colors.get(status, (128, 128, 128))
    
    # Main info box
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (10, 10), (350, 160), color, 2)
    
    # Display based on status
    if letter and status == 'valid':
        cv2.putText(frame, letter, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
        cv2.putText(frame, f"{confidence:.0%}", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    elif status == 'nonsense':
        cv2.putText(frame, "NONSENSE", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, "Not a valid sign", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    elif status == 'uncertain':
        cv2.putText(frame, "Uncertain", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Entropy: {entropy:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    elif status == 'stabilizing':
        cv2.putText(frame, "Hold still...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(frame, "No hand", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, "Show your hand", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    
    # Top 3 predictions
    if top3:
        y = 130
        text = f"Top: {top3[0][0]}({top3[0][1]:.0%}) {top3[1][0]}({top3[1][1]:.0%}) {top3[2][0]}({top3[2][1]:.0%})"
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    
    # Info bar (bottom)
    cv2.rectangle(frame, (0, h-30), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, f"Model: {model_name} | FPS: {fps:.0f} | ESC=quit R=reset D=debug", 
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    
    # Entropy meter (right side)
    if entropy > 0:
        bar_h = int(min(entropy / 3.0, 1.0) * 100)
        bar_color = (0, 255, 0) if entropy < 1.0 else (0, 255, 255) if entropy < 2.0 else (0, 0, 255)
        cv2.rectangle(frame, (w-30, 170), (w-10, 170+100), (50,50,50), -1)
        cv2.rectangle(frame, (w-30, 170+100-bar_h), (w-10, 170+100), bar_color, -1)
        cv2.putText(frame, "E", (w-25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)


def main():
    print("=" * 60)
    print("SIGN LANGUAGE INFERENCE")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    model, model_name, test_acc = load_model(MODEL_PATH, device)
    print(f"✓ Model: {model_name} (Test acc: {test_acc:.2f}%)")
    
    # Load encoder
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    print(f"✓ Classes: {list(le.classes_)}")
    
    # MediaPipe
    print("Initializing MediaPipe...")
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    smoother = PredictionSmoother(
        window_size=SMOOTHING_WINDOW,
        min_agreement=MIN_AGREEMENT,
        switch_threshold=SWITCH_THRESHOLD
    )
    smoother.min_frames_before_switch = MIN_FRAMES_BEFORE_SWITCH
    
    # Webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  ESC - Quit")
    print("  R   - Reset buffer")
    print("  S   - Save frame + landmarks")
    print("  D   - Toggle debug info")
    print("=" * 60)
    print("\nStarting...\n")
    
    frame_count = 0
    hand_frames = 0
    show_debug = False
    fps = 0
    fps_timer = cv2.getTickCount()
    letter_counts = Counter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS calculation
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / ((cv2.getTickCount() - fps_timer) / cv2.getTickFrequency())
            fps_timer = cv2.getTickCount()
        
        # Process frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        letter = None
        confidence = 0
        entropy = 0
        top3 = None
        status = 'no_hand'
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            hand_frames += 1
            
            # Draw landmarks (mirrored)
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                s = hand.landmark[connection[0]]
                e = hand.landmark[connection[1]]
                sp = (int((1-s.x)*w), int(s.y*h))
                ep = (int((1-e.x)*w), int(e.y*h))
                cv2.line(frame, sp, ep, (0, 200, 0), 2)
            
            for i, lm in enumerate(hand.landmark):
                cx, cy = int((1-lm.x)*w), int(lm.y*h)
                color = (0, 0, 255) if i == 0 else (255, 100, 0) if i == 9 else (255, 0, 0)
                cv2.circle(frame, (cx, cy), 4, color, -1)
            
            # Wait for stable hand
            if hand_frames < MIN_HAND_FRAMES:
                status = 'stabilizing'
                progress = hand_frames / MIN_HAND_FRAMES
                cv2.rectangle(frame, (10, 170), (10 + int(200*progress), 180), (255,255,0), -1)
            else:
                # Extract and normalize landmarks
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
                normalized = normalize_landmarks(coords)
                
                # Predict
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(normalized).unsqueeze(0).to(device)
                    outputs = model(x_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                pred_idx = probs.argmax()
                pred_letter = le.classes_[pred_idx]
                confidence = probs[pred_idx]
                entropy = calculate_entropy(probs)
                
                # Top 3
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3 = [(le.classes_[i], probs[i]) for i in top3_idx]
                
                # Debug output
                if show_debug and frame_count % 15 == 0:
                    print(f"[{frame_count}] {top3[0][0]}({top3[0][1]:.2f}) {top3[1][0]}({top3[1][1]:.2f}) | entropy={entropy:.2f}")
                
                # Smoothing
                if entropy < ENTROPY_THRESHOLD and confidence > CONFIDENCE_THRESHOLD:
                    smoother.add(pred_idx, confidence)
                    smoothed, avg_conf = smoother.get_smoothed()
                    
                    if smoothed is not None:
                        letter = le.classes_[smoothed]
                        confidence = avg_conf
                        
                        if letter == "Nonsense":
                            status = 'nonsense'
                        else:
                            status = 'valid'
                            letter_counts[letter] += 1
                    else:
                        status = 'stabilizing'
                else:
                    smoother.reset()
                    status = 'uncertain'
        else:
            hand_frames = 0
            smoother.reset()
        
        # Draw UI
        draw_ui(frame, letter, confidence, entropy, top3, status, model_name, fps)
        
        # Debug overlay
        if show_debug:
            y = 200
            cv2.putText(frame, f"Hand frames: {hand_frames}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            y += 20
            cv2.putText(frame, f"Buffer: {len(smoother.buffer)}/{SMOOTHING_WINDOW}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            y += 20
            sticky = le.classes_[smoother.current_letter] if smoother.current_letter is not None else "None"
            cv2.putText(frame, f"Sticky: {sticky} ({smoother.frames_since_switch} frames)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            y += 20
            if letter_counts:
                top_letters = letter_counts.most_common(5)
                cv2.putText(frame, f"Session: {dict(top_letters)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        cv2.imshow('Sign Language Recognition', frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            smoother.reset()
            hand_frames = 0
            print("Reset!")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug: {'ON' if show_debug else 'OFF'}")
        elif key == ord('s') and results.multi_hand_landmarks:
            import time
            ts = int(time.time())
            cv2.imwrite(f"frame_{ts}.png", frame)
            np.save(f"landmarks_{ts}.npy", normalized)
            print(f"Saved frame_{ts}.png + landmarks_{ts}.npy")
            print(f"  Prediction: {pred_letter} ({confidence:.2f})")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Session stats
    print("\n" + "=" * 60)
    print("SESSION STATS")
    print("=" * 60)
    if letter_counts:
        print("Letters detected:")
        for letter, count in letter_counts.most_common():
            print(f"  {letter}: {count}")
    print("\nDone!")


if __name__ == "__main__":
    main()