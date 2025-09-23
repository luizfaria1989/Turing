from layers.base import Layer
import numpy as np


class HardTanh(Layer):
    """Implements the Hard Tanh activation layer.

    The Hard Tanh is a computationally cheaper, piecewise linear
    approximation of the standard Tanh function. Its formula is
    f(x) = max(-1, min(1, x)).

    It is faster to compute than Tanh and avoids the use of exponentials.

    Attributes:
        input (np.ndarray): Stores the input from the forward pass, which is
            needed to calculate the gradient in the backward pass.
    """

    def __init__(self):
        """Initializes the HardTanh activation layer."""
        super().__init__()
        self.input = None

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data from the previous layer.

        Returns:
            np.ndarray: The output of the layer after applying the Hard Tanh function.
        """
        self.input = input_data
        # This is a more concise way to write the piecewise function
        return np.clip(self.input, -1, 1)

    def backward(self, grad_output):
        """Performs the backward pass of the layer.

        The derivative of Hard Tanh is 1 in the linear region between -1
        and 1, and 0 otherwise.

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to
                this layer's output.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient of the loss with respect to this layer's input.
                - None, since Hard Tanh has no trainable parameters.
        """
        # Create a gradient mask that is 1 where the input was between -1 and 1
        hard_tanh_grad = np.where((self.input > -1) & (self.input < 1), 1, 0)

        return grad_output * hard_tanh_grad, None