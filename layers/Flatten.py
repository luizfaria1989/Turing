import numpy as np
from layers.base import Layer


class Flatten(Layer):
    """A layer that flattens a multi-dimensional input into a 2D tensor.

    This layer is typically used to transition from convolutional/pooling layers
    (which output multi-dimensional feature maps) to dense layers (which expect
    a 2D input of shape (batch_size, features)).

    Attributes:
        input_shape (tuple): Stores the shape of the input from the forward
            pass, which is needed to un-flatten the gradient in the
            backward pass.
    """

    def __init__(self):
        """Initializes the Dense layer."""
        super().__init__()
        self.input_shape = None

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data, typically with a shape like
                (batch_size, height, width, channels).

        Returns:
            np.ndarray: The flattened output with shape (batch_size, -1), where -1
                is the product of all other dimensions (height * width * channels).
        """
        # Cache the original input shape for the backward pass
        self.input_shape = input_data.shape
        # Flatten the input while preserving the batch dimension
        flatten_output = input_data.reshape(input_data.shape[0], -1)

        return flatten_output

    def backward(self, output_gradient):
        """Performs the backward pass of the layer.

        It reshapes the incoming gradient back to the original input shape.

        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect
                to the output of this layer.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The reshaped gradient with respect to this layer's input.
                - None, since Flatten has no trainable parameters.
        """
        # Reshape the gradient back to the original input shape
        input_gradient = output_gradient.reshape(self.input_shape)
        return input_gradient, None