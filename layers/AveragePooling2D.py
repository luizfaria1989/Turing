import numpy as np
from layers.base import Layer

class AveragePooling2D(Layer):
    """Implements an AveragePooling layer for 2D inputs (e.g., images).

    AveragePooling is a downsampling operation that reduces the spatial
    dimensions (height, width) of the input by taking the average value
    over a defined window.

    Attributes:
        pool_size (tuple): The size of the window to average pool over.
        stride (tuple): The step size of the pooling window.
    """

    def __init__(self, pool_size=(2, 2), stride=None):
        """Initializes the AveragePooling2D layer.

        Args:
            pool_size (tuple, optional): A tuple of 2 integers (height, width)
                specifying the size of the window to average pool over.
                Defaults to (2, 2).
            stride (int or tuple, optional): The step size of the pooling window.
                If None, the stride will default to the pool size.
                Defaults to None.
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

    def forward(self, input_data):
        """Performs the forward pass of the AveragePooling layer.

        Args:
            input_data (np.ndarray): The input data with shape
                (batch_size, input_height, input_width, channels).

        Returns:
            np.ndarray: The downsampled output with shape
                (batch_size, output_height, output_width, channels).
        """
        (batches, input_height, input_width, channels) = input_data.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride

        output_height = int((input_height - pool_h) / stride_h) + 1
        output_width = int((input_width - pool_w) / stride_w) + 1

        output_matrix = np.zeros((batches, output_height, output_width, channels))

        for b in range(batches):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        start_h = h * stride_h
                        start_w = w * stride_w
                        end_h = start_h + pool_h
                        end_w = start_w + pool_w

                        pooling_window = input_data[b, start_h:end_h, start_w:end_w, c]
                        output_matrix[b, h, w, c] = np.mean(pooling_window)  # Changed to mean

        return output_matrix

    def backward(self, output_gradient):
        """Performs the backward pass of the AveragePooling layer.

        Distributes the incoming gradient equally among all the elements
        in the original pooling window.

        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect
                to the output of this layer.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient with respect to this layer's input.
                - None, since this layer has no trainable parameters.
        """
        pool_h, pool_w = self.pool_size

        # Calculate the value of the distributed gradient
        distributed_grad = output_gradient / (pool_h * pool_w)

        # Upsample the distributed gradient to the size of the original input
        upsampled_grad = np.repeat(distributed_grad, pool_h, axis=1)
        upsampled_grad = np.repeat(upsampled_grad, pool_w, axis=2)

        return upsampled_grad, None