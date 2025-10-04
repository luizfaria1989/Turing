import numpy as np
from optimizers.base import Optimizer

class RMSProp(Optimizer):
    """Implements the RMSProp optimizer.

    The update formulas are:
    1. n_t = v * n_{t-1} + (1 - v) * g_t^2
    2. θ_t = θ_{t-1} - η * g_t / sqrt(n_t + ε)

    Attributes:
        lr (float): The learning rate (η).
        decay_rate (float): The decay rate for the moving average (v).
        epsilon (float): A small constant for numerical stability (ε).
        accumulator (list): The 'memory' that stores the moving average of
                            squared gradients.
    """
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-7):
        """Initializes the RMSProp optimizer.

        Args:
            learning_rate (float, optional): The learning rate (η).
                Defaults to 0.001.
            decay_rate (float, optional): The decay rate for the moving
                average (v). Defaults to 0.9.
            epsilon (float, optional): A small value for numerical stability (ε).
                Defaults to 1e-7.
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.accumulator = None

    def step(self, params, grads):
        """Performs a single optimization step for all parameters.

        Args:
            params (list): A list of lists containing the model's parameters.
                Expected structure: [[weights1, biases1], [weights2, biases2], ...]
            grads (list): A list of lists containing the gradients for each
                parameter, with the same structure as `params`.
        """
        if self.accumulator is None:
            self.accumulator = [[np.zeros_like(p) for p in group] for group in params]

        # Iterate over parameters for each layer
        for i in range(len(params)):
            # --- Weight Updates ---
            # 1. Update accumulator with moving average: n_t = v*n_{t-1} + (1-v)*g_t^2
            self.accumulator[i][0] = self.decay_rate * self.accumulator[i][0] + \
                                     (1 - self.decay_rate) * grads[i][0] ** 2
            # 2. Calculate and apply update: θ_t = θ_{t-1} - η * g_t / sqrt(n_t + ε)
            params[i][0] -= self.lr * grads[i][0] / (np.sqrt(self.accumulator[i][0]) + self.epsilon)

            # --- Bias Updates ---
            # 1. Update accumulator with moving average
            self.accumulator[i][1] = self.decay_rate * self.accumulator[i][1] + \
                                     (1 - self.decay_rate) * grads[i][1] ** 2
            # 2. Calculate and apply update
            params[i][1] -= self.lr * grads[i][1] / (np.sqrt(self.accumulator[i][1]) + self.epsilon)