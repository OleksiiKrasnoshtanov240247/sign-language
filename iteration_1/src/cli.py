import cv2
import time
import sys
import os
import torch
from pathlib import Path

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.backend.detector import SignDetector, is_dynamic

# Model path
CHECKPOINT_PATH = os.path.join(project_root, "models", "static", "best_model.pth")


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
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}.")
    
    try:
        detector = SignDetector(static_model_path=ckpt_path, device=device)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return

    # Try to open webcam with DirectShow (better for Windows)
    if os.name == 'nt':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    target_letter = 'A'
    last_prediction = None  # For static letters
    dynamic_result = None   # For dynamic letters (J/Z)
    dynamic_result_time = 0 # When dynamic result was received
    no_hand_frames = 0
    
    # FPS Control
    TARGET_FPS = 15
    interval = 1.0 / TARGET_FPS
    last_capture = 0
    
    print("Controls:")
    print("  A-Z: Change target letter")
    print("  SPACE: Start recording (for J/Z dynamic letters)")
    print("  ESC: Quit")
    
    try:
        while True:
            now = time.time()
            if now - last_capture < interval:
                time.sleep(0.001) 
                continue
            last_capture = now
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            frame = cv2.flip(frame, 1)
            pass_frame = frame.copy() 
            
            landmarks, status, data, prediction_info = detector.process_frame(pass_frame, target_letter)
            
            # Handle prediction display
            if landmarks is None:
                no_hand_frames += 1
                if no_hand_frames > 10 and not is_dynamic(target_letter):
                    last_prediction = None
            else:
                no_hand_frames = 0
                
                if is_dynamic(target_letter):
                    # Dynamic letters - only update when we get new result
                    if prediction_info is not None:
                        dynamic_result = prediction_info
                        dynamic_result_time = time.time()
                        print(f"✅ Result: {prediction_info.get('predicted_letter')} ({prediction_info.get('confidence', 0):.2f})")
                    # Clear after 10 seconds
                    elif dynamic_result and (time.time() - dynamic_result_time > 10):
                        dynamic_result = None
                else:
                    # Static letters - always update
                    if prediction_info is not None:
                        last_prediction = prediction_info
            
            # Visualization
            detector.hand_capture.visualize_landmarks(frame)
            
            # Status Overlay
            if landmarks is not None:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            
            if last_prediction and "[MATCH]" in status:
                color = (0, 255, 0)

            # Override status for dynamic result persistence
            if is_dynamic(target_letter) and dynamic_result:
                pred_letter = dynamic_result.get('predicted_letter')
                conf = dynamic_result.get('confidence', 0)
                match = "[MATCH]" if pred_letter == target_letter else "[NO MATCH]"
                status = f"Letter: {target_letter} | Hand: YES | Result: {pred_letter} ({conf:.2f}) {match}"
                color = (0, 255, 0) # Green for result
                
                # Also update last_prediction so probabilities are shown
                last_prediction = dynamic_result
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"FPS Limit: {TARGET_FPS} | Key: {target_letter}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            mode = "DYNAMIC" if is_dynamic(target_letter) else "STATIC"
            hint = "SPACE: Start Recording" if is_dynamic(target_letter) else ""
            cv2.putText(frame, f"Mode: {mode} | {hint} | ESC: Quit", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show probabilities only if we have a valid prediction AND hand is visible
            if last_prediction and landmarks is not None and last_prediction.get('all_probabilities'):
                sorted_probs = sorted(last_prediction['all_probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
                y_offset = 110
                for letter, prob in sorted_probs:
                    text_color = (0, 255, 255)
                    if letter == target_letter:
                        text_color = (0, 255, 0)
                    cv2.putText(frame, f"{letter}: {prob:.2f}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    y_offset += 20
            
            cv2.imshow("Sign Language Detector", frame)
            
            # Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):
                # SPACE - start recording for dynamic letters
                if is_dynamic(target_letter):
                    detector.start_recording()
                    last_prediction = None
                    dynamic_result = None
                    dynamic_result_time = 0
                    print(f"🎬 Recording {target_letter}...")
                else:
                    detector.clear_buffer()
                    last_prediction = None
                    print("Buffer cleared")
            elif ord('a') <= key <= ord('z'):
                target_letter = chr(key).upper()
                detector.clear_buffer()
                last_prediction = None
                dynamic_result = None
                dynamic_result_time = 0
                print(f"Target: {target_letter}")
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Detector closed.")


if __name__ == "__main__":
    run_app()