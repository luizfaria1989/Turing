import numpy as np
# from layers.base import Layer # Assuming you have a base class

class Sequential:
    """A container for stacking neural network layers in a sequence.

    This class manages a list of layers and orchestrates the forward pass of
    data and the backward pass of gradients.

    Attributes:
        layers (list): The list of layer objects that make up the network.
        params (list): A list of lists containing the trainable parameters
            (weights and biases) from all layers in the network that have them.
        grads (list): A list of lists containing the gradients for each
            parameter. This list is populated after each call to `backward`.
    """
    def __init__(self, layers):
        """Initializes the Sequential model.

        Args:
            layers (list): A list of layer objects (e.g., [Dense(), ReLU()])
                in the order they should be applied.
        """
        self.layers = layers
        self.params = []
        self.grads = []

        # Smart loop that only collects parameters from layers that actually have them
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                self.params.append(layer.params)

    def forward(self, input_data):
        """Performs the forward pass for the entire network.

        Args:
            input_data (np.ndarray): The input data for the first layer.

        Returns:
            np.ndarray: The final output of the network after passing through all layers.
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_gradient):
        """Performs the backward pass (backpropagation) for the entire network.

        Args:
            output_gradient (np.ndarray): The initial gradient from the loss function
                with respect to the network's output.

        Returns:
            list: A list of lists containing the gradients for all trainable
                parameters in the network.
        """
        self.grads = []
        grad = output_gradient

        # Iterate over the layers in reverse order
        for layer in reversed(self.layers):
            grad, layer_grads = layer.backward(grad)

            # Collect gradients only from layers that returned them
            if layer_grads:
                self.grads.insert(0, layer_grads)

        return self.grads