import numpy as np
from src.backend.video.capture import HandCapture, normalize

# J and Z need movement, rest are static
DYNAMIC_LETTERS = ['J', 'Z']

class SignDetector:
    def __init__(self):
        self.hand_capture = HandCapture()
        self.buffer = []
        
    def is_dynamic(self, letter):
        return letter.upper() in DYNAMIC_LETTERS

    def process_frame(self, frame, target_letter):
        """
        Process a frame and return separation of concerns:
        - landmarks: raw landmarks or None
        - status: text description of status
        - data: prepared data for model (if ready) or None
        """
        landmarks = self.hand_capture.extract_landmarks(frame)
        
        status = f"Letter: {target_letter} | Hand: NO"
        data = None
        
        if landmarks is not None:
            norm = normalize(landmarks)
            status = f"Letter: {target_letter} | Hand: YES"
            
            if self.is_dynamic(target_letter):
                self.buffer.append(norm)
                if len(self.buffer) > 30:
                    self.buffer.pop(0)
                
                status += f" | Buffer: {len(self.buffer)}/30"
                
                if len(self.buffer) == 30:
                    # Shape (30, 21, 3) -> flattened or kept as is depending on model expectation
                    # For now returning as (30, 21, 3) as per plan description "Ready: (30, ...)"
                    # converting to numpy array
                    data_array = np.array(self.buffer)
                    data = data_array # .reshape(30, -1) if needed flattened
                    status += f" | Ready: {data.shape}"
            else:
                # Static letter
                self.buffer = [] # clear buffer if switched mode
                data = norm.flatten()
                status += f" | Ready: {data.shape}"
        else:
            # If hand lost, maybe clear buffer? User snippet didn't clear explicitly but implied reset on target change
            pass
            
        return landmarks, status, data
    
    def clear_buffer(self):
        self.buffer = []
