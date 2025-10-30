from adafruit_servokit import ServoKit
import time

class PanTilt:
    def __init__(self):
        self.kit = ServoKit(channels=16)
        self.pan_ch, self.tilt_ch = 0, 1
        self.kit.servo[self.pan_ch].angle = 90
        self.kit.servo[self.tilt_ch].angle = 90
    def set_pan(self, angle):
        self.kit.servo[self.pan_ch].angle = max(0,min(180,angle))
    def set_tilt(self, angle):
        self.kit.servo[self.tilt_ch].angle = max(0,min(180,angle))
