import numpy as np
from layers.base import Layer  # Assuming your base class is here


class Convolution2D(Layer):
    """Implements a 2D Convolutional layer.

    This layer creates a convolution kernel that is convolved with the layer
    input to produce a tensor of outputs (feature maps). It is the core
    building block for Convolutional Neural Networks (CNNs).

    Attributes:
        kernels (np.ndarray): The weights (filters) of the layer.
        biases (np.ndarray): The bias vector of the layer.
        params (list): A list containing the trainable parameters [kernels, biases].
        stride (tuple): The stride of the convolution.
        padding (int): The amount of zero-padding added to the input.
    """

    def __init__(self, num_filters, kernel_size, input_channels, stride=1, padding=0):
        """Initializes the Convolution2D layer.

        Args:
            num_filters (int): The number of filters the layer will learn. This
                determines the number of output channels.
            kernel_size (tuple): A tuple of 2 integers for kernel height and width.
            input_channels (int): The number of channels in the input image.
            stride (int or tuple, optional): The stride of the convolution. Defaults to 1.
            padding (int, optional): The amount of zero-padding. Defaults to 0.
        """
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding

        # Initialize weights (kernels) and biases
        kernel_h, kernel_w = self.kernel_size
        self.kernels = np.random.randn(num_filters, kernel_h, kernel_w, self.input_channels) * 0.01
        self.biases = np.zeros((1, 1, 1, num_filters))
        self.params = [self.kernels, self.biases]
        self.input_padded = None

    def forward(self, input_data):
        """Performs the forward pass of the convolution.

        Args:
            input_data (np.ndarray): Input data with shape (N, H_in, W_in, C_in).

        Returns:
            np.ndarray: Output feature map with shape (N, H_out, W_out, F).
        """
        (batch_size, input_height, input_width, _) = input_data.shape
        (filters, kernel_height, kernel_width, _) = self.kernels.shape
        stride_height, stride_width = self.stride

        # Apply padding
        pad_config = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        self.input_padded = np.pad(input_data, pad_config, mode='constant')

        # Calculate output dimensions
        output_height = int((input_height - kernel_height + 2 * self.padding) / stride_height) + 1
        output_width = int((input_width - kernel_width + 2 * self.padding) / stride_width) + 1

        # Create output matrix
        output_matrix = np.zeros((batch_size, output_height, output_width, self.num_filters))

        # Perform convolution
        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(output_height):
                    for w in range(output_width):
                        start_h, start_w = h * stride_height, w * stride_width
                        end_h, end_w = start_h + kernel_height, start_w + kernel_width

                        window = self.input_padded[b, start_h:end_h, start_w:end_w, :]

                        conv_sum = np.sum(window * self.kernels[f, ...])
                        output_matrix[b, h, w, f] = conv_sum + self.biases[0, 0, 0, f]

        return output_matrix

    def backward(self, output_gradient):
        """Performs the backward pass (backpropagation) for the convolution.

        Args:
            output_gradient (np.ndarray): Gradient of the loss with respect to
                the output of this layer.

        Returns:
            tuple[np.ndarray, list]: A tuple containing:
                - The gradient with respect to the layer's input.
                - A list with the gradients for the kernels and biases.
        """
        (filters, kernel_height, kernel_width, _) = self.kernels.shape
        (batch_size, _, _, _) = output_gradient.shape
        (_, output_height, output_width, _) = output_gradient.shape
        stride_height, stride_width = self.stride

        d_kernels = np.zeros_like(self.kernels)
        d_biases = np.zeros_like(self.biases)
        d_input_padded = np.zeros_like(self.input_padded)

        for b in range(batch_size):
            for f in range(filters):
                for h in range(output_height):
                    for w in range(output_width):
                        start_h, start_w = h * stride_height, w * stride_width
                        end_h, end_w = start_h + kernel_height, start_w + kernel_width

                        window = self.input_padded[b, start_h:end_h, start_w:end_w, :]
                        grad_pixel = output_gradient[b, h, w, f]

                        # Accumulate gradients
                        d_kernels[f, ...] += window * grad_pixel
                        d_biases[0, 0, 0, f] += grad_pixel
                        d_input_padded[b, start_h:end_h, start_w:end_w, :] += self.kernels[f, ...] * grad_pixel

        # Remove padding from the input gradient
        if self.padding > 0:
            p = self.padding
            d_input = d_input_padded[:, p:-p, p:-p, :]
        else:
            d_input = d_input_padded

        return d_input, [d_kernels, d_biases]