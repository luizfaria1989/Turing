import numpy as np
from layers.base import Layer

class Softmax(Layer):
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.probs

    def backward(self, grad_output):
        # Garante 2D com batch dimension correto
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)  # 1 amostra
        elif grad_output.shape[0] != self.probs.shape[0]:
            grad_output = grad_output.reshape(self.probs.shape)  # batch dimension
        return grad_output

