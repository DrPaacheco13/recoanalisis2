import numpy as np
from ie_module import Module  # Asumiendo que 'ie_module' es un módulo que proporciona la integración con OpenVINO
from utils import resize_input  # Asumiendo que 'resize_input' es una función de utilidad para redimensionar la entrada
from openvino.runtime import PartialShape

class AgeGenderDetector(Module):
    class Result:
        OUTPUT_SIZE = 4

        def __init__(self, output):
            self.age = int(output[0])
            self.gender = "Male" if output[1] < 0.5 else "Female"  # Umbral de 0.5 para género
            self.confidence_age = output[2]
            self.confidence_gender = output[3]

    def __init__(self, core, model, input_size, confidence_threshold=0.5):
        super(AgeGenderDetector, self).__init__(core, model, 'Age and Gender Detection')
        if len(self.model.inputs) != 1:
            raise RuntimeError("El modelo debe tener exactamente 1 entrada para Age and Gender Detection")

        # Usamos la primera salida si hay más de una
        if len(self.model.outputs) < 1:
            raise RuntimeError("El modelo debe tener al menos 1 salida para Age and Gender Detection")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        if input_size[0] > 0 and input_size[1] > 0:
            self.model.reshape({self.input_tensor_name: PartialShape([1, 3, *input_size])})
        elif not (input_size[0] == 0 and input_size[1] == 0):
            raise ValueError("Ambas dimensiones de la entrada deben ser positivas para Age and Gender Detector")

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        self.output_shape = self.model.outputs[0].shape

        if len(self.output_shape) != 4:
            raise RuntimeError("El modelo debe tener una forma de salida con {} salidas".format(self.Result.OUTPUT_SIZE))

        if confidence_threshold > 1.0 or confidence_threshold < 0:
            raise ValueError("El umbral de confianza debe estar en el rango [0; 1]")

        self.confidence_threshold = confidence_threshold

    def preprocess(self, frame):
        return resize_input(frame, self.input_shape, self.nchw_layout)

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        super(AgeGenderDetector, self).enqueue({self.input_tensor_name: input})

    def postprocess(self):
        outputs = self.get_outputs()[0]
        results = []
        for output in outputs[0][0]:
            result = AgeGenderDetector.Result(output)
            if result.confidence_age < self.confidence_threshold:
                break  # Resultados ordenados por confianza descendente

            results.append(result)

        return results
