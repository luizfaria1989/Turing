import numpy as np
from layers.base import Layer # Assuming your base class is here

class Softmax(Layer):
    """Implements the Softmax activation layer.

    Softmax is typically used as the final activation layer in multi-class
    classification problems. It transforms a vector of scores (logits) into a
    probability distribution, where each value is in the range [0, 1] and all
    values sum to 1.

    Attributes:
        probs (np.ndarray): Stores the output probabilities from the forward pass.
    """
    def __init__(self):
        """Initializes the Softmax layer."""
        super().__init__()
        self.probs = None

    def forward(self, x):
        """Performs the forward pass of the layer.

        Args:
            x (np.ndarray): The input data (logits) from the previous layer,
                with shape (batch_size, num_classes).

        Returns:
            np.ndarray: The output probabilities, with the same shape as the input.
        """
        # Subtracting the max value is a trick for numerical stability,
        # preventing overflow with large input values.
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.probs

    def backward(self, grad_output):
        """Performs the backward pass of the layer.

        IMPORTANT NOTE: This is a simplified backward pass that is ONLY valid
        when Softmax is used in conjunction with the Categorical Cross-Entropy
        loss function. The complex Jacobian of the Softmax is conveniently
        canceled out by the derivative of the Cross-Entropy loss, resulting in
        a simple pass-through of the gradient.

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to
                this layer's output. In our design, this will be (y_pred - y_true).

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient with respect to the layer's input.
                - None, since Softmax has no trainable parameters.
        """
        # The gradient simply passes through the layer.
        return grad_output, None