import numpy as np
from loss.base import LossFunction  # Assuming your base class is here


class CategoricalCrossEntropy(LossFunction):
    """Implements the Categorical Cross-Entropy loss function.

    This is the standard loss function for multi-class classification problems.
    It measures the dissimilarity between the predicted probability distribution
    (from Softmax) and the true distribution (the one-hot encoded labels).
    """

    def __init__(self):
        """Initializes the loss function."""
        super().__init__()

    def forward(self, y_pred, y_true):
        """Calculates the mean Categorical Cross-Entropy loss for a batch.

        Args:
            y_pred (np.ndarray): The predicted probability distributions from the
                network, with shape (batch_size, num_classes).
            y_true (np.ndarray): The true labels in one-hot encoded format, with
                the same shape as y_pred.

        Returns:
            float: The mean loss value for the batch.
        """
        # Number of samples in the batch
        m = y_true.shape[0]
        # Clip predictions to avoid log(0), which would result in nan error
        y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)

        # Calculate the total loss and normalize it by the batch size
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        return loss

    def backward(self, y_pred, y_true):
        """Calculates the initial gradient of the loss for backpropagation.

        NOTE: This is the simplified derivative of the combined Softmax and
        Cross-Entropy functions, which makes the initial gradient calculation
        very efficient.

        Args:
            y_pred (np.ndarray): The predicted probability distributions.
            y_true (np.ndarray): The true labels in one-hot encoded format.

        Returns:
            np.ndarray: The gradient of the loss with respect to the network's
                output, ready to be passed to the last layer's backward method.
        """
        # Normalize the gradient by the batch size
        m = y_true.shape[0]
        grad_output = (y_pred - y_true) / m
        return grad_output