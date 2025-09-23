from optimizers.base import Optimizer # Assuming your base class is here

class GradientDescent(Optimizer):
    """Implements the classic Gradient Descent (GD) optimization algorithm.

    This class inherits from the base Optimizer class and implements the
    parameter update logic following the classic formula:
    θ_new = θ_old - α * ∇J(θ)

    Attributes:
        lr (float): The learning rate (α) used for the parameter updates.
    """
    def __init__(self, learning_rate=0.001):
        """Initializes the GradientDescent optimizer.

        Args:
            learning_rate (float, optional): The learning rate (α).
                Defaults to 0.001.
        """
        super().__init__(learning_rate)

    def step(self, params, grads):
        """Performs a single optimization step for all parameters.

        Args:
            params (list): A list of lists containing the model's parameters.
                Expected structure: [[weights1, biases1], [weights2, biases2], ...]
            grads (list): A list of lists containing the gradients for each
                parameter, with the same structure as `params`.
        """
        # Iterate over each parameter group (one per Dense layer)
        for i in range(len(params)):
            # The formula: parameter -= learning_rate * parameter_gradient
            params[i][0] -= self.lr * grads[i][0]  # Update weights (w)
            params[i][1] -= self.lr * grads[i][1]  # Update biases (b)