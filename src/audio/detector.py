import tflite_runtime.interpreter as tflite
import numpy as np, librosa

class AudioBirdDetector:
    def __init__(self, model_path):
        self.interp = tflite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def infer(self, y, sr):
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        logmel = librosa.power_to_db(mels).astype(np.float32)
        x = np.expand_dims(logmel, (0, -1))
        self.interp.set_tensor(self.inp['index'], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out['index'])[0]
