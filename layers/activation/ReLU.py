import numpy as np
from layers.base import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, grad_output):
        grad_input = grad_output * (self.input > 0)
        return grad_input
