import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.params = []
        self.grads = []

        for layer in self.layers:
            if hasattr(layer, 'params'):
                self.params.extend(layer.params)

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        self.grads = []
        grad_input = grad_output
        for layer in reversed(self.layers):
            if hasattr(layer, 'params'):
                layer_back = layer.backward(grad_input)
                grad_input = layer_back[0]  # sem np.atleast_2d
                self.grads.extend(layer_back[1:])
            else:
                grad_input = layer.backward(grad_input)
            print(f"Depois de {layer.__class__.__name__}.backward, grad_input.shape: {grad_input.shape}")
        return self.params, self.grads
