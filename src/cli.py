import cv2
import time
import sys
import os
import torch
from pathlib import Path

# Fix import path to allow running this script directly
# Add the project root (two levels up) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.backend.detector import SignDetector

# Updated to the found model path
# Use relative path from project root or absolute path
CHECKPOINT_PATH = os.path.join(project_root, "CNN_model", "best_model.pth")

def run_app():
    print("Initializing Sign Language Detector...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Check if checkpoint exists
    ckpt_path = None
    if Path(CHECKPOINT_PATH).exists():
        ckpt_path = CHECKPOINT_PATH
        print(f"Found model at: {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Models will not verify signs.")
    
    try:
        detector = SignDetector(static_model_path=ckpt_path, device=device)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    target_letter = 'A'
    
    # FPS Control
    TARGET_FPS = 15
    interval = 1.0 / TARGET_FPS
    last_capture = 0
    
    print("Controls:")
    print("  A-Z: Change target letter")
    print("  Q: Quit")
    
    try:
        while True:
            # Time-based throttling
            now = time.time()
            if now - last_capture < interval:
                time.sleep(0.001) 
                continue
                
            last_capture = now
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            pass_frame = frame.copy() 
            landmarks, status, data, prediction_info = detector.process_frame(pass_frame, target_letter)
            
            # Visualization
            detector.hand_capture.visualize_landmarks(frame)
            
            # Status Overlay
            color = (0, 255, 0) if "Hand: YES" in status else (0, 0, 255)
            if "[MATCH]" in status:
                 color = (0, 255, 0) # Green for match
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"FPS Limit: {TARGET_FPS} | Key: {target_letter}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "Press 'Q' to quit", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show Detailed Probabilities
            if prediction_info and prediction_info.get('all_probabilities'):
                # Display top 3
                sorted_probs = sorted(prediction_info['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
                y_offset = 110
                for letter, prob in sorted_probs:
                    text_color = (0, 255, 255) # Yellow
                    if letter == target_letter:
                        text_color = (0, 255, 0) # Green if it matches target
                        
                    cv2.putText(frame, f"{letter}: {prob:.2f}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    y_offset += 20
            
            cv2.imshow("Sign Language Detector", frame)
            
            # Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # 'q' to quit
                break
            elif ord('a') <= key <= ord('z'):
                target_letter = chr(key).upper()
                detector.clear_buffer() # Reset buffer on change
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Detector closed.")

if __name__ == "__main__":
    run_app()
