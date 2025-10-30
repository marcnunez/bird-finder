import pyaudio, numpy as np

class AudioStream:
    def __init__(self, rate=32000, chunk=1024, channels=4):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=chunk)
        self.channels, self.rate = channels, rate

    def read(self):
        data = self.stream.read(self.stream._frames_per_buffer, exception_on_overflow=False)
        arr = np.frombuffer(data, dtype=np.float32)
        return arr.reshape(-1, self.channels)
