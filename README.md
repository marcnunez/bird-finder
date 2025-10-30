# bird-finder

## Multimodal bird detection, localization, and classification for edge devices

Bird-finder is a lightweight Python-based system that detects, localizes, and identifies birds in real time on embedded platforms such as the Raspberry Pi.
It combines audio and visual processing to autonomously capture and recognize birds in their natural environment.

## Features

- Audio detection: Real-time microphone stream analysis to detect bird songs using compact TensorFlow Lite models.

- Sound localization: Direction-of-arrival estimation using a multi-microphone array and GCC-PHAT.

- Visual detection: OpenCV-based camera interface with YOLO-nano for bird bounding box detection.

- Pan-tilt tracking: Automatically orients the camera toward the detected sound source.

- Species classification: EfficientNet-Lite or MobileNet models for classifying bird species from both audio and image data.

- Edge-ready: Optimized for Raspberry Pi 4/5 and Coral TPU accelerators.

- Data logging: Stores audio clips, images, and metadata for later analysis or model retraining.

## Tech Stack

- Languages: Python 3, C++ (for low-latency audio)

- Libraries: OpenCV, NumPy, PyAudio, TensorFlow Lite, Adafruit ServoKit

- Hardware: Raspberry Pi 4/5, Pi Camera V2 or HQ, 2- or 4-Microphone Array, PCA9685 Pan-Tilt Mount

