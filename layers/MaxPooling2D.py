import numpy as np
from layers.base import Layer

class MaxPooling2D(Layer):
    """Implements a MaxPooling layer for 2D inputs (e.g., images).

     MaxPooling is a downsampling operation that reduces the spatial dimensions
     (height, width) of the input, helping to make the model more efficient and
     invariant to small translations in the input features.

     Attributes:
         pool_size (tuple): The size of the window to max pool over.
         stride (tuple): The step size of the pooling window.
         cache (np.ndarray): Stores a mask indicating the location of the max
             values from the forward pass, needed for backpropagation.
     """

    def __init__(self, pool_size=(2,2), stride=None, mode='vectorized'):
        """Initializes the MaxPooling2D layer.

        Args:
            pool_size (tuple, optional): A tuple of 2 integers (height, width)
                specifying the size of the window to max pool over.
                Defaults to (2, 2).
            stride (int or tuple, optional): The step size of the pooling window.
                If None, the stride will default to the pool size.
                Defaults to None.
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.cache = None
        self.mode = mode

    def forward(self, input_data):
        """Dispatches to the appropriate forward implementation based on mode"""
        if self.mode == 'naive':
            return self._forward_naive(input_data)
        elif self.mode == 'vectorized':
            return self._forward_vectorized(input_data)
        else:
            raise ValueError("Invalid mode. Choose 'naive' or 'vectorized'.")

    def backward(self, output_gradient):
        """Dispatches to the appropriate backward implementation based on mode."""
        if self.mode == 'naive':
            return self._backward_naive(output_gradient)
        elif self.mode == 'vectorized':
            return self._backward_vectorized(output_gradient)
        else:
            raise ValueError("Invalid mode. Choose 'naive' or 'vectorized'.")

    def _forward_naive(self, input_data):
        """Performs the forward pass of the MaxPooling layer.

        Args:
            input_data (np.ndarray): The input data with shape
                (batch_size, input_height, input_width, channels).

        Returns:
            np.ndarray: The downsampled output with shape
                (batch_size, output_height, output_width, channels).
        """

        (batches, input_height, input_width, channels) = input_data.shape

        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride

        # Calculates the output height and width
        output_height = int((input_height - pool_height) / stride_height) + 1
        output_width = int((input_width - pool_width) / stride_width) + 1

        # Create the output matrix and the cache for the backward pass
        output_matrix = np.zeros((batches, output_height, output_width, channels))
        self.cache = np.zeros_like(input_data)

        for b in range(batches):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):

                        # Define the limits of the current window
                        start_height = h * stride_height
                        start_width = w * stride_width
                        end_height = start_height + pool_height
                        end_width = start_width + pool_width

                        # Slice the window from the input
                        pooling_window = input_data[b, start_height:end_height, start_width:end_width, c]

                        # Find the maximum value
                        max_arg_value = np.max(pooling_window)

                        # Place the max value in the output matrix
                        output_matrix[b, h, w, c] = max_arg_value

                        # Create a mask to mark the location of the max value
                        mask = (pooling_window == max_arg_value)

                        # Store the mask in the cache
                        self.cache[b, start_height:end_height, start_width:end_width, c] = mask

        return output_matrix

    def _backward_naive(self, output_gradient):
        """Performs the backward pass of the MaxPooling layer.

        Distributes the incoming gradient to the locations that had the maximum
        value during the forward pass.

        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect
                to the output of this layer.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient with respect to this layer's input.
        """

        upsampled_grad = np.repeat(output_gradient, self.pool_size[0], axis=1)
        upsampled_grad = np.repeat(upsampled_grad, self.pool_size[1], axis=2)

        grad_input = upsampled_grad * self.cache

        return grad_input, None

    def _forward_vectorized(self, input_data):
        """Performs the forward pass of the MaxPooling layer.

        Args:
            input_data (np.ndarray): The input data with shape
                (batch_size, input_height, input_width, channels).

        Returns:
            np.ndarray: The downsampled output with shape
                (batch_size, output_height, output_width, channels).
        """

        (batch_size, input_height, input_width, channels) = input_data.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride

        output_height = int((input_height - pool_height) / stride_height) + 1
        output_width = int((input_width - pool_width) / stride_width) + 1

        output_matrix = np.zeros((batch_size, output_height, output_width, channels))
        self.cache = np.zeros_like(input_data)

        for n in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        start_height, start_width = h * stride_height, w * stride_width
                        end_h, end_w = start_height + pool_height, start_width + pool_width

                        window = input_data[n, start_height:end_h, start_width:end_w, c]
                        max_value = np.max(window)
                        output_matrix[n, h, w, c] = max_value

                        mask = (window == max_value)
                        self.cache[n, start_height:end_h, start_width:end_w, c] = mask

        return output_matrix

    def _backward_vectorized(self, output_gradient):
        """Performs the backward pass of the MaxPooling layer.

        Distributes the incoming gradient to the locations that had the maximum
        value during the forward pass.

        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect
                to the output of this layer.

        Returns:
            tuple[np.ndarray, None]: A tuple containing:
                - The gradient with respect to this layer's input.
        """

        # "Stretches" the output gradient to the size of the original input
        upsampled_grad = np.repeat(output_gradient, self.pool_size[0], axis=1)
        upsampled_grad = np.repeat(upsampled_grad, self.pool_size[1], axis=2)

        # Multiplies by the mask to pass the gradient only to the correct locations
        input_grad = upsampled_grad * self.cache

        return input_grad, None