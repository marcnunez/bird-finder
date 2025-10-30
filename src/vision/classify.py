import tflite_runtime.interpreter as tflite
import numpy as np, cv2

class BirdClassifier:
    def __init__(self, model):
        self.interp = tflite.Interpreter(model_path=model)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def infer(self, crop):
        crop = cv2.resize(crop, (224,224))
        crop = np.expand_dims(crop.astype(np.float32)/255., 0)
        self.interp.set_tensor(self.inp['index'], crop)
        self.interp.invoke()
        return self.interp.get_tensor(self.out['index'])[0]
