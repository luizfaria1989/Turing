import numpy as np
from loss.base import LossFunction

class CategoricalCrossEntropy(LossFunction):
    def forward(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(np.clip(y_pred, 1e-9, 1-1e-9))) / m
        return loss

    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / y_true.shape[0]
