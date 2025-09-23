from layers.base import Layer
import numpy as np

class HardSigmoid(Layer):
    """Implements the Hard Sigmoid activation layer.

    The Hard Sigmoid is a computationally cheaper, piecewise linear
    approximation of the standard Sigmoid function. Its formula is
    f(x) = max(0, min(1, x / 6 + 0.5)).

    It is faster to compute than Sigmoid and can be useful in specific
    architectures, especially in resource-constrained environments.

    Attributes:
        input (np.ndarray): Stores the input from the forward pass, which is
            needed to calculate the gradient in the backward pass.
    """

    def __init__(self):
        """Initializes the HardSigmoid activation layer."""
        super().__init__()
        self.input = None

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data from the previous layer.

        Returns:
            np.ndarray: The output of the layer after applying the Hard Sigmoid function.
        """
        self.input = input_data

        # This is an efficient way to write max(0, min(1, x / 6 + 0.5))
        output = self.input / 6 + 0.5
        output = np.clip(output, 0, 1)  # A more concise way to handle the bounds

        return output

    def backward(self, grad_output):
        """Performs the backward pass of the layer.

        The derivative of Hard Sigmoid is a constant (1/6) in the linear
        region between -3 and 3, and 0 otherwise.

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to
                this layer's output.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient of the loss with respect to this layer's input.
                - None, since Hard Sigmoid has no trainable parameters.
        """
        # Create a gradient mask that is 1/6 everywhere
        hard_sigmoid_grad = np.full_like(self.input, 1 / 6)

        # Set the gradient to 0 for the flat regions
        hard_sigmoid_grad[self.input < -3] = 0
        hard_sigmoid_grad[self.input > 3] = 0

        return grad_output * hard_sigmoid_grad, None



