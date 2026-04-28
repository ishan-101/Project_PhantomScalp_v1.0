import numpy as np
import onnxruntime as ort

class ONNXModel:
    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict_proba(self, features):
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        proba = self.sess.run([self.output_name], {self.input_name: x})[0]
        return float(proba[0][1])