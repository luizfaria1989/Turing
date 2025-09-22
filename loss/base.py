class LossFunction:

    def __init__(self):
        pass

    def forward(self, y_pred, y_real):
        raise NotImplementedError()

    def backward(self, y_pred, y_real):
        raise NotImplementedError()
