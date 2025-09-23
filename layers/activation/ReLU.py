import numpy as np
from layers.base import Layer  # Assuming your base class is here


class ReLU(Layer):
    """Implements the ReLU (Rectified Linear Unit) activation layer.

    The Rectified Linear Unit is a non-linear function with the formula
    f(x) = max(0, x). It is one of the most common activation functions used
    in neural networks because it is computationally efficient and helps
    mitigate the vanishing gradient problem.

    Attributes:
        input (np.ndarray): Stores the input from the forward pass, which is
            needed to calculate the gradient in the backward pass.
    """

    def __init__(self):
        """Initializes the ReLU activation layer."""
        super().__init__()
        self.input = None

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data from the previous layer.

        Returns:
            np.ndarray: The output of the layer after applying ReLU.
        """
        # Caches the input for the backward pass
        self.input = input_data
        return np.maximum(0, self.input)

    def backward(self, grad_output):
        """Performs the backward pass of the layer.

        It computes the derivative of the ReLU function and applies it to the
        incoming gradient from the next layer (using the chain rule).

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to
                this layer's output.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient of the loss with respect to this layer's input.
                - None, since ReLU has no trainable parameters.
        """
        # The derivative of ReLU is 1 for inputs > 0, and 0 otherwise.
        relu_grad = (self.input > 0)

        # Apply the chain rule
        return grad_output * relu_grad, None