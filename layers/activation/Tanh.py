import numpy as np
from layers.base import Layer  # Assuming your base class is here

class Tanh(Layer):
    """Implements the Tanh (Hyperbolic Tangent) activation layer.

    The Tanh function squashes any input value into the range [-1, 1].
    Because its output is zero-centered, it can sometimes lead to faster
    convergence than the Sigmoid function, but can still suffer from the
    vanishing gradient problem in deep networks.

    Attributes:
        tanh_output (np.ndarray): Stores the output from the forward pass, which
            is needed to efficiently calculate the gradient in the backward pass.
    """

    def __init__(self):
        """Initializes the Tanh activation layer."""
        super().__init__()
        self.input = None
        self.tanh_output = None

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data from the previous layer.

        Returns:
            np.ndarray: The output of the layer after applying Tanh.
        """
        # Caches the input for the backward pass
        self.input = input_data
        self.tanh_output = np.tanh(self.input)
        return self.tanh_output

    def backward(self, grad_output):
        """Performs the backward pass of the layer.

                It computes the derivative of the Tanh function using its output and
                applies it to the incoming gradient from the next layer.

                Args:
                    grad_output (np.ndarray): The gradient of the loss with respect to
                        this layer's output.

                Returns:
                    tuple[np.ndarray, None]: A tuple containing:
                        - The gradient of the loss with respect to this layer's input.
                        - None, since Tanh has no trainable parameters.
                """
        # The derivative of Tanh is 1 - tanh(x)**2
        tanh_grad = 1 - self.tanh_output**2

        # Apply the chain rule
        return grad_output * tanh_grad, None