import numpy as np
from layers.base import Layer  # Assuming your base class is here


class Sigmoid(Layer):
    """Implements the Sigmoid activation layer.

    The Sigmoid function squashes any input value into the range [0, 1].
    It was historically common in hidden layers but is now more often used in
    the output layer for binary classification problems.

    Attributes:
        sigmoid_output (np.ndarray): Stores the output from the forward pass,
            which is needed to efficiently calculate the gradient in the
            backward pass.
    """

    def __init__(self):
        """Initializes the Sigmoid activation layer."""
        super().__init__()
        self.input = None
        self.sigmoid = None

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data from the previous layer.

        Returns:
            np.ndarray: The output of the layer after applying Sigmoid.
        """
        # Caches the input for the backward pass
        self.input = input_data
        self.sigmoid_output = 1/ (1 + np.exp(-input_data))
        return self.sigmoid_output

    def backward(self, grad_output):
        """Performs the backward pass of the layer.

          It computes the derivative of the Sigmoid function using its output,
          and applies it to the incoming gradient from the next layer.

          Args:
              grad_output (np.ndarray): The gradient of the loss with respect to
                  this layer's output.

          Returns:
              tuple[np.ndarray, None]: A tuple containing:
                  - The gradient of the loss with respect to this layer's input.
                  - None, since Sigmoid has no trainable parameters.
          """
        # The derivative is sigmoid(x) * (1 - sigmoid(x))
        sigmoid_grad = self.sigmoid_output * (1 - self.sigmoid_output)

        # Apply the chain rule
        return grad_output * sigmoid_grad, None