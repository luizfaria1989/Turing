class Layer:

    def __init__(self, input, output):
        self.input = input
        self.output = output

    def forward(self, input):
        NotImplementedError()

    def backward(self, grad_input, learning_rate):
        NotImplementedError()
