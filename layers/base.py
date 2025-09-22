class Layer:

    def __init__(self):
        self.params = []
        self.grad = []

    def forward(self, input):
        raise NotImplementedError()

    def backward(self, grad_input):
        raise NotImplementedError()
