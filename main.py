import cv2
import time
import sys
from src.backend.detector import SignDetector

def main():
    print("Initializing Sign Language Detector...")
    
    detector = SignDetector()
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
                # Sleep briefly to avoid busy wait, or just continue
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
            pass_frame = frame.copy() # preserve original if needed
            landmarks, status, data = detector.process_frame(pass_frame, target_letter)
            
            # Visualization
            detector.hand_capture.visualize_landmarks(frame)
            
            # Status Overlay
            color = (0, 255, 0) if "Hand: YES" in status else (0, 0, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"FPS Limit: {TARGET_FPS} | Key: {target_letter}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Sign Language Detector", frame)
            
            # Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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
    main()
