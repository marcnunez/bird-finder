import cv2, numpy as np

class BirdDetector:
    def __init__(self, onnx_path, conf=0.4):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.conf = conf
    def infer(self, img):
        blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), swapRB=True)
        self.net.setInput(blob)
        preds = self.net.forward()[0]
        boxes=[]
        for p in preds:
            conf = p[4]
            if conf>self.conf:
                x,y,w,h = p[:4]
                boxes.append([x,y,w,h,conf])
        return boxes
