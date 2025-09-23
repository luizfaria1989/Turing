import numpy as np
from layers.base import Layer  # Assuming your base class is here


class LeakyReLU(Layer):
    """Implements the Leaky ReLU (Leaky Rectified Linear Unit) activation layer.

    The Leaky ReLU is a variation of the ReLU function. For positive inputs, it
    behaves like a standard ReLU (f(x) = x), but for negative inputs, it has a
    small, non-zero, constant slope (f(x) = alpha * x). This helps to prevent
    the "Dying ReLU" problem.

    Attributes:
        input (np.ndarray): Stores the input from the forward pass, which is
            needed to calculate the gradient in the backward pass.
        alpha (float): The slope for negative input values.
    """

    def __init__(self, alpha=0.01):
        """Initializes the LeakyReLU activation layer.

        Args:
            alpha (float, optional): The slope for negative inputs. Defaults to 0.01.
        """
        super().__init__()
        self.input = None
        self.alpha = alpha

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data from the previous layer.

        Returns:
            np.ndarray: The output of the layer after applying ReLU.
        """
        # Caches the input for the backward pass
        self.input = input_data
        return np.maximum(self.input * self.alpha, self.input)

    def backward(self, grad_output):
        """Performs the backward pass of the layer.

        It computes the derivative of the Leaky ReLU function and applies it to
        the incoming gradient from the next layer (using the chain rule).

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to
                this layer's output.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient of the loss with respect to this layer's input.
                - None, since LeakyReLU has no trainable parameters.
        """
        # The derivative is 1 for inputs > 0, and alpha otherwise.
        leaky_relu_grad = np.where(self.input > 0, 1, self.alpha)

        # Apply the chain rule
        return grad_output * leaky_relu_grad, None