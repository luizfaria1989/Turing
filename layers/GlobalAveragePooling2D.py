import numpy as np
from layers.base import Layer

class GlobalAveragePooling2D(Layer):
    """Implements a Global Average Pooling layer for 2D inputs.

    This layer computes the average of the entire spatial dimensions (height,
    width) for each feature map (channel). It transforms an input of shape
    (batch_size, height, width, channels) into an output of shape
    (batch_size, 1, 1, channels). It is often used as an alternative to a
    Flatten layer before the final dense layers in a CNN.

    Attributes:
        input_shape (tuple): Stores the shape of the input from the forward
            pass, needed for the backward pass.
    """

    def __init__(self):
        """Initializes the GlobalAveragePooling2D layer."""
        super().__init__()
        self.input_shape = None

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data with shape
                (batch_size, height, width, channels).

        Returns:
            np.ndarray: The output with shape (batch_size, 1, 1, channels).
        """
        # Cache the original input shape for the backward pass
        self.input_shape = input_data.shape

        # Cache the original input shape for the backward pass
        output = np.mean(input_data, axis=(1, 2), keepdims=True)

        return output

    def backward(self, output_gradient):
        """Performs the backward pass of the layer.

        Distributes the incoming gradient equally to all spatial locations
        of the original input.

        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect
                to the output of this layer.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient with respect to this layer's input.
                - None, since this layer has no trainable parameters.
        """
        # Get the original spatial dimensions from the cached shape
        _, input_h, input_w, _ = self.input_shape

        # Calculate the value of the distributed gradient
        distributed_grad = output_gradient / (input_h * input_w)

        # Upsample the distributed gradient to the size of the original input
        # by creating an array of ones and multiplying (leveraging broadcasting)
        upsampled_grad = np.ones(self.input_shape) * distributed_grad

        return upsampled_grad, None