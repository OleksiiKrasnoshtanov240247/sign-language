import cv2

class VideoCapture:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open a camera")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        flag, frame = self.cap.read()
        if not flag:
            raise RuntimeError("Failed to read frame from the source.")
        return frame
    
    def release(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.release()

if __name__ == "__main__":
    video_capture = VideoCapture(0)
    try:
        while True:
            frame = video_capture.read()
            cv2.imshow('Video Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()