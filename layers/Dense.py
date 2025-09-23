import numpy as np
from layers.base import Layer


class Dense(Layer):
    """Implements a fully connected (dense) neural network layer.

    A dense layer performs a linear transformation on the input data according
    to the equation Y = XW + b. It is a fundamental building block for
    most neural networks.

    Attributes:
        w (np.ndarray): The weight matrix of the layer.
        b (np.ndarray): The bias vector of the layer.
        params (list): A list containing the layer's trainable parameters, [w, b].
        activation (Layer, optional): The activation layer to be applied after
            the linear transformation.
    """

    def __init__(self, input_size, output_size, activation=None):
        """Initializes the Dense layer.

        Args:
            input_size (int): The number of input features (neurons from the previous layer).
            output_size (int): The number of output neurons for this layer.
            activation (Layer, optional): An activation layer to be used. Defaults to None.
        """
        super().__init__()
        self.w = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.input = None
        self.params = [self.w, self.b]
        self.activation = activation

    def forward(self, input_data):
        """Performs the forward pass of the layer.

        Args:
            input_data (np.ndarray): The input data with shape (batch_size, input_size).

        Returns:
            np.ndarray: The output from the layer with shape (batch_size, output_size).
        """
        self.input = input_data
        # Perform the linear transformation
        linear_output = self.input @ self.w + self.b

        # Apply the activation function, if it exists
        if self.activation:
            return self.activation.forward(linear_output)

        return linear_output

    def backward(self, output_gradient):
        """Performs the backward pass (backpropagation) through the layer.

        Calculates the gradients of the loss with respect to the layer's
        inputs, weights, and biases.

        Args:
            output_gradient (np.ndarray): The gradient of the loss with respect
                to the output of this layer.

        Returns:
            tuple[np.ndarray, list]: A tuple containing:
                - The gradient of the loss with respect to the layer's input.
                - A list containing the gradients for the weights and biases.
        """
        # If an activation function exists, first perform its backward pass
        if self.activation:
            output_gradient = self.activation.backward(output_gradient)

        # Gradients for the parameters (weights and biases)
        grad_w = self.input.T @ output_gradient
        grad_b = np.sum(output_gradient, axis=0, keepdims=True)

        # Gradient to pass to the previous layer
        input_gradient = output_gradient @ self.w.T

        return input_gradient, [grad_w, grad_b]