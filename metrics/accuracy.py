import numpy as np


class Accuracy():
    def __init__(self):
        pass

    def calculate(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)
        correct_predictions = np.argmax(y_true, axis=1)
        correct = np.sum(predictions == correct_predictions)
        accuracy = correct / len(predictions)
        return accuracy
