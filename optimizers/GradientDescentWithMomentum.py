from optimizers.base import Optimizer
import numpy as np

class GradientDescentWithMomentum(Optimizer):
    """Implements the Gradient Descent with Momentum optimizer.

    This optimizer accelerates convergence by adding 'inertia' (velocity)
    to the parameter updates. It 'remembers' the direction of the last
    step, which helps navigate ravines in the loss landscape and avoid
    shallow local minima.

    The update formulas are:
    1. v_t = momentum * v_{t-1} - learning_rate * g_t
    2. θ_t = θ_{t-1} + v_t

    Attributes:
        lr (float): The learning rate (ε).
        momentum (float): The momentum factor (α) that controls inertia.
        velocity (list): The 'memory' that stores the velocity for each parameter.
    """
    def __init__(self, momentum=0.9, learning_rate=0.001):
        """Initializes the Momentum optimizer.

        Args:
            learning_rate (float, optional): The learning rate (ε).
                Defaults to 0.001.
            momentum (float, optional): The momentum factor (α), typically
                close to 1. Defaults to 0.9.
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
            # 1. Calculate new velocity: v_t = α*v_{t-1} - ε*g_t
            self.velocity[i][0] = self.momentum * self.velocity[i][0] - self.lr * grads[i][0]
            params[i][0] += self.velocity[i][0]

            # --- Bias Updates ---
            # 1. Calculate new velocity
            self.velocity[i][1] = self.momentum * self.velocity[i][1] - self.lr * grads[i][1]
            params[i][1] += self.velocity[i][1]