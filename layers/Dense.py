import numpy as np
from layers.base import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.w = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros((1, output_size))
        self.input = None
        self.params = [self.w, self.b]

    def forward(self, input_data):
        self.input = input_data
        return input_data @ self.w + self.b

    def backward(self, output_gradient):
        print("Antes de Dense.backward")
        print("grad_input.shape:", output_gradient.shape)
        output_gradient = np.atleast_2d(output_gradient)  # força 2D

        grad_input = output_gradient @ self.w.T
        grad_w = self.input.T @ output_gradient
        grad_b = np.sum(output_gradient, axis=0, keepdims=True)
        return grad_input, grad_w, grad_b
