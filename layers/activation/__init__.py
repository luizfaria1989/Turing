from layers.base import Layer

class ActivationFunction(Layer):
    def __init__(self, function, prime):
        self.function = function
        self.prime = prime
        super().__init__()

    def forward(self, input):
        self.input = input
        output = self.function(input)
        return output

    def backward(self, learning_rate, grad_input):
        grad_output = grad_input * self.prime(self.input)
        return grad_output
