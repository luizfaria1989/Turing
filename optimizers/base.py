class Optimizer():

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def step(self, parameters, gradients):
        raise NotImplementedError
