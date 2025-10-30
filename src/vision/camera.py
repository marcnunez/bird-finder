import cv2
class Camera:
    def __init__(self, idx=0, w=640, h=480):
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None
