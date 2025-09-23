from layers.base import Layer
import numpy as np

class Softsign(Layer):
    """Implements the Softsign activation layer.

    The Softsign function is a non-linear activation function that squashes
    input values into the range [-1, 1], similar to the Tanh function.
    Its formula is f(x) = x / (1 + |x|). It can be an alternative to Tanh
    but is less prone to saturation.

    Attributes:
        input (np.ndarray): Stores the input from the forward pass, which is
            needed to calculate the gradient in the backward pass.
    """

    def __init__(self):
        """Initializes the Softsign activation layer."""
        super().__init__()
        self.input = None

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data from the previous layer.

        Returns:
            np.ndarray: The output of the layer after applying the Softsign function.
        """
        self.input = input_data
        return self.input / (1 + np.abs(self.input))

    def backward(self, grad_output):
        """Performs the backward pass of the layer.

        It computes the derivative of the Softsign function and applies it to the
        incoming gradient from the next layer (using the chain rule).

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to
                this layer's output.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient of the loss with respect to this layer's input.
                - None, since Softsign has no trainable parameters.
        """
        # The derivative of Softsign is 1 / (1 + |x|)**2
        softsign_grad =  (1 / (1 + np.abs(self.input))**2)

        # Apply the chain rule
        return grad_output * softsign_grad, None