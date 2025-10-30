# main.py
import asyncio, time
from audio.capture import AudioStream
from audio.doa import doa_azimuth
from audio.detector import AudioBirdDetector
from vision.camera import Camera
from vision.detect import BirdDetector
from vision.classify import BirdClassifier
from control.pantilt import PanTilt

async def main():
    audio = AudioStream(channels=2)
    aud_model = AudioBirdDetector("models/bird_presence.tflite")
    cam = Camera()
    det = BirdDetector("models/bird_detector.onnx")
    clf = BirdClassifier("models/bird_classifier.tflite")
    pt = PanTilt()

    while True:
        frame = cam.read()
        mics = audio.read()
        angle = doa_azimuth(mics)
        # combine channels for inference
        y = mics.mean(axis=1)
        bird_prob = aud_model.infer(y, audio.rate)[0]
        if bird_prob > 0.7:
            print(f"Bird sound detected, angle≈{angle:.1f}°")
            pt.set_pan(90 + angle)      # map to servo angle
            await asyncio.sleep(0.4)
            frame = cam.read()
            boxes = det.infer(frame)
            if boxes:
                x,y,w,h,_ = max(boxes, key=lambda b:b[4])
                crop = frame[int(y):int(y+h), int(x):int(x+w)]
                species = clf.infer(crop)
                print("Species prediction:", species.argmax())
        await asyncio.sleep(0.1)

asyncio.run(main())
