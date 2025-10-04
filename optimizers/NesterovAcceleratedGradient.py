from optimizers.base import Optimizer
import numpy as np

class NesterovAcceleratedGradient(Optimizer):
    """Implements the Nesterov Accelerated Gradient (NAG) optimizer.

    NAG is an improvement over standard Momentum. It is 'smarter' because
    it calculates the gradient after a 'lookahead' step in the direction of the
    momentum. This results in more responsive updates and often faster
    convergence.

    The update formulas for this implementation are:
    1. v_t = momentum * v_{t-1} - learning_rate * g_t
    2. θ_t = θ_{t-1} + momentum * v_t - learning_rate * g_t

    Attributes:
        lr (float): The learning rate (ε).
        momentum (float): The momentum factor (μ) that controls inertia.
        velocity (list): The 'memory' that stores the velocity for each parameter.
    """
    def __init__(self, momentum=0.9, learning_rate=0.001):
        """Initializes the NAG optimizer.

        Args:
            learning_rate (float, optional): The learning rate (ε).
                Defaults to 0.001.
            momentum (float, optional): The momentum factor (μ).
                Defaults to 0.9.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        """Performs a single optimization step for all parameters.

        Args:
            params (list): A list of lists containing the model's parameters.
                Expected structure: [[weights1, biases1], [weights2, biases2], ...]
            grads (list): A list of lists containing the gradients for each
                parameter, with the same structure as `params`.
        """

        if self.velocity is None:
            self.velocity = [[np.zeros_like(p) for p in group] for group in params]

        # Iterate over each parameter group (one per Dense layer)
        for i in range(len(params)):
            # --- Weight Updates ---
            # 1. Calculate new velocity (same as standard momentum)
            self.velocity[i][0] = self.momentum * self.velocity[i][0] - self.lr * grads[i][0]
            # 2. Apply NAG update: θ_t = θ_{t-1} + μ*v_t - ε*g_t
            params[i][0] += self.momentum * self.velocity[i][0] - self.lr * grads[i][0]

            # --- Bias Updates ---
            # 1. Calculate new velocity
            self.velocity[i][1] = self.momentum * self.velocity[i][1] - self.lr * grads[i][1]
            # 2. Apply NAG update
            params[i][1] += self.momentum * self.velocity[i][1] - self.lr * grads[i][1]