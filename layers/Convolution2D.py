import numpy as np
from layers.base import Layer
from utils.im2col import im2col_indices, col2im_indices

class Convolution2D(Layer):
    """Implements a 2D Convolutional layer with both a naive and vectorized version.

    This layer creates convolution kernels that are convolved with the layer input
    to produce a tensor of outputs (feature maps). It is the core building block
    for Convolutional Neural Networks (CNNs).

    The implementation can be chosen via the `mode` parameter.
    - 'naive': Uses nested Python for-loops. Easy to understand but very slow.
    - 'vectorized': Uses an im2col transformation to perform convolution via a
      single, highly efficient matrix multiplication.

    Attributes:
        kernels (np.ndarray): The weights (filters) of the layer.
        biases (np.ndarray): The bias vector of the layer.
        params (list): A list containing the trainable parameters [kernels, biases].
        mode (str): The implementation mode ('naive' or 'vectorized').
    """

    def __init__(self, input_channels, num_filters, kernel_size, stride=1, padding=0, mode='vectorized'):
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding
        self.mode = mode

        # Initialize weights (kernels) and biases
        # Kernels are initialized in (F, C, K_h, K_w) format for im2col compatibility
        kernel_height, kernel_width = self.kernel_size
        self.kernels = np.random.randn(num_filters, input_channels, kernel_height, kernel_width) * 0.01
        self.biases = np.zeros((num_filters, 1))
        self.params = [self.kernels, self.biases]
        self.cache = None

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

    # --- NAIVE (LOOP-BASED) IMPLEMENTATION ---
    def _forward_naive(self, input_data):
        """Performs a naive forward pass using nested for-loops..

        Args:
            input_data (np.ndarray): Input data with shape (batch_size, input_height, input_width, input_channels).

        Returns:
            np.ndarray: Output feature map with shape (N, H_out, W_out, F).
        """

        (batch_size, input_height, input_width, input_channels) = input_data.shape
        filters, _, kernel_height, kernel_width = self.kernels.shape
        stride_height, stride_width = self.stride

        # Apply padding
        pad_config = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        input_padded = np.pad(input_data, pad_config, mode='constant')
        self.cache = input_padded

        # Calculate output dimensions
        output_height = int((input_height - kernel_height + 2 * self.padding) / stride_height) + 1
        output_width = int((input_width - kernel_width + 2 * self.padding) / stride_width) + 1

        # Create output matrix
        output_matrix = np.zeros((batch_size, output_height, output_width, filters))

        # Perform convolution
        for n in range(batch_size):
            for f in range(filters):
                for h in range(output_height):
                    for w in range(output_width):
                        start_height, start_width = h * stride_height, w * stride_width
                        end_height, end_width = start_height + kernel_height, start_width + kernel_width

                        window = input_padded[n, start_height:end_height, start_width:end_width, :]
                        current_kernel = self.kernels[f, ...].transpose(1, 2, 0)

                        conv_sum = np.sum(window * current_kernel)
                        output_matrix[n, h, w, f] = conv_sum + self.biases[f]

        return output_matrix

    def _backward_naive(self, output_gradient):
        """Performs a naive backward pass using nested for-loops.

        Args:
            output_gradient (np.ndarray): Gradient of the loss with respect to
                the output of this layer.

        Returns:
            tuple[np.ndarray, list]: A tuple containing:
                - The gradient with respect to the layer's input.
                - A list with the gradients for the kernels and biases.
        """

        input_padded = self.cache
        (filters, input_channels, kernel_height, kernel_width) = self.kernels.shape
        (batch_size, output_height, output_width, _) = output_gradient.shape
        stride_height, stride_width = self.stride

        grad_kernels = np.zeros_like(self.kernels)
        grad_biases = np.zeros_like(self.biases)
        grad_input_padded = np.zeros_like(input_padded)

        for n in range(batch_size):
            for f in range(filters):
                for h in range(output_height):
                    for w in range(output_width):
                        start_height, start_width = h * stride_height, w * stride_width
                        end_height, end_width = start_height + kernel_height, start_width + kernel_width

                        window = input_padded[n, start_height:end_height, start_width:end_width, :]
                        grad_pixel = output_gradient[n, h, w, f]

                        # Accumulate gradients
                        grad_kernels[f, ...] += window.transpose(2, 0, 1) * grad_pixel
                        grad_biases[f] += grad_pixel
                        grad_input_padded[n, start_height:end_height, start_width:end_width, :] += self.kernels[f, ...].transpose(1, 2,
                                                                                                             0) * grad_pixel
        # Remove padding from the input gradient
        if self.padding > 0:
            p = self.padding
            grad_input = grad_input_padded[:, p:-p, p:-p, :]
        else:
            grad_input = grad_input_padded
        return grad_input, [grad_kernels, grad_biases]

    # --- VECTORIZED (IM2COL-BASED) IMPLEMENTATION ---
    def _forward_vectorized(self, input_data):
        """Performs a fast forward pass using im2col transformation.

        Args:
            output_gradient (np.ndarray): Gradient of the loss with respect to
                the output of this layer.

        Returns:
            tuple[np.ndarray, list]: A tuple containing:
                - The gradient with respect to the layer's input.
                - A list with the gradients for the kernels and biases.
        """

        batch_size, input_height, input_width, input_channels = input_data.shape
        filters, _, kernel_height, kernel_width = self.kernels.shape
        stride_height, _ = self.stride

        # Calculate output dimensions
        output_height = int((input_height - kernel_height + 2 * self.padding) / stride_height) + 1
        output_width = int((input_width - kernel_width + 2 * self.padding) / stride_height) + 1

        # Convert from NHWC (batch, height, width, channels) (library standard)
        # to NCHW (batch, channels, height, width) (im2col standard)
        transposed_input_data = input_data.transpose(0, 3, 1, 2)

        # Unroll image windows into columns
        input_col = im2col_indices(transposed_input_data, kernel_height, kernel_width, padding=self.padding, stride=stride_height)
        weights_row = self.kernels.reshape(filters, -1)

        # The core convolution is now a single matrix multiplication
        output_matrix = weights_row @ input_col + self.biases

        # Reshape the output back to image format
        output_matrix = output_matrix.reshape(filters, output_height, output_width, batch_size)

        # Convert the output back to the library's standard NHWC format
        output_matrix = output_matrix.transpose(3, 1, 2, 0)

        # Cache necessary variables for the backward pass
        self.cache = (input_data, input_col)

        return output_matrix

    def _backward_vectorized(self, output_gradient):
        """Performs a fast backward pass using col2im transformation.

         Args:
             output_gradient (np.ndarray): Gradient of the loss with respect to
                 the output of this layer.

         Returns:
             tuple[np.ndarray, list]: A tuple containing:
                 - The gradient with respect to the layer's input.
                 - A list with the gradients for the kernels and biases.
         """

        input_data, input_data_col = self.cache
        filters, channels, kernel_height, kernel_width = self.kernels.shape

        # Gradient for biases is the sum of the output gradient over all but the filter dimension
        grad_biases = np.sum(output_gradient, axis=(0, 1, 2)).reshape(-1, 1)

        # Reshape output gradient for matrix multiplication
        output_gradient_transposed = output_gradient.transpose(3, 1, 2, 0).reshape(filters, -1)

        # Calculate gradient for kernels (flattened)
        grad_kernels_flat = output_gradient_transposed @ input_data_col.T
        grad_kernels = grad_kernels_flat.reshape(self.kernels.shape)

        # Calculate gradient for the input (in column format)
        weights_row = self.kernels.reshape(filters, -1)
        grad_input_col = weights_row.T @ output_gradient_transposed

        # Reconstruct input gradient from columns back to image format (NCHW)
        input_nchw_shape = (input_data.shape[0], input_data.shape[3], input_data.shape[1], input_data.shape[2])
        grad_input_nchw = col2im_indices(grad_input_col, input_nchw_shape, kernel_height, kernel_width, padding=self.padding, stride=self.stride[0])
        grad_input = grad_input_nchw.transpose(0, 2, 3, 1)

        return grad_input, [grad_kernels, grad_biases]