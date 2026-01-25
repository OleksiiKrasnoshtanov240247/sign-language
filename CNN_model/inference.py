import cv2
import mediapipe as mp
import torch
import numpy as np
from model import ASLClassifier
import config

# Load Model & Labels
classes = np.load(config.LABEL_ENCODER_PATH, allow_pickle=True)
model = ASLClassifier(config.INPUT_SIZE, config.NUM_CLASSES).to(config.DEVICE)
model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
model.eval()

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            data_points = []
            for lm in hand_lms.landmark:
                data_points.extend([lm.x, lm.y, lm.z])

            input_tensor = torch.tensor(data_points, dtype=torch.float32).to(config.DEVICE).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                confidence = torch.softmax(output, dim=1).max().item()
                prediction = classes[output.argmax(dim=1).item()]

            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            if confidence > config.CONFIDENCE_THRESHOLD:
                cv2.putText(
                    frame,
                    f"{prediction} ({confidence:.2f})",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("NGT Real-time Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()