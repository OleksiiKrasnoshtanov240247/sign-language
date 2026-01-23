"""
NGT Hand Detector
Target letter → MediaPipe → Normalize
"""

import cv2
import numpy as np
import mediapipe as mp


# J and Z need movement, rest are static
DYNAMIC_LETTERS = ['J', 'Z']


def is_dynamic(letter):
    return letter.upper() in DYNAMIC_LETTERS


def normalize(landmarks):
    """Center on wrist + scale by hand size"""
    lm = landmarks.copy()
    lm -= lm[0]  # center on wrist
    scale = np.linalg.norm(lm[12])  # distance to middle finger
    if scale > 0:
        lm /= scale
    return lm


class HandDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def detect(self, frame):
        """Returns (21,3) landmarks or None"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        if not result.multi_hand_landmarks:
            return None
        
        hand = result.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
    
    def close(self):
        self.hands.close()


# Test
if __name__ == "__main__":
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    target = 'A'
    buffer = []  # for dynamic letters
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        landmarks = detector.detect(frame)
        
        # Status text
        if landmarks is not None:
            norm = normalize(landmarks)
            status = f"Letter: {target} | Hand: YES"
            color = (0, 255, 0)
            
            if is_dynamic(target):
                buffer.append(norm)
                if len(buffer) > 30:
                    buffer.pop(0)
                status += f" | Buffer: {len(buffer)}/30"
                if len(buffer) == 30:
                    data = np.array(buffer).reshape(30, -1)
                    status += f" | Ready: {data.shape}"
            else:
                data = norm.flatten()
                status += f" | Ready: {data.shape}"
        else:
            status = f"Letter: {target} | Hand: NO"
            color = (0, 0, 255)
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, "Keys: A-Z change letter, Q quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('a') <= key <= ord('z'):
            target = chr(key).upper()
            buffer = []
    
    detector.close()
    cap.release()
    cv2.destroyAllWindows()