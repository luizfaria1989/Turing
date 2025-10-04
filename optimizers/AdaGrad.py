import numpy as np
from optimizers.base import Optimizer

class AdaGrad(Optimizer):
    """Implements the Adaptive Gradient (AdaGrad) optimizer.

    AdaGrad adapts the learning rate for each parameter individually, performing
    larger updates for infrequent and smaller updates for frequent parameters.
    It is particularly well-suited for sparse data. Its main weakness is that
    the learning rate aggressively decays and may stop learning too early.

    The update formulas are:
    1. n_t = n_{t-1} + g_t^2
    2. θ_t = θ_{t-1} - η * g_t / sqrt(n_t + ε)

    Attributes:
        lr (float): The learning rate (η).
        epsilon (float): A small constant for numerical stability (ε).
        accumulator (list): The 'memory' that stores the sum of squared gradients.
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-7):
        """Initializes the AdaGrad optimizer.

        Args:
            learning_rate (float, optional): The learning rate (η).
                Defaults to 0.01.
            epsilon (float, optional): A small value for numerical stability (ε)
                to prevent division by zero. Defaults to 1e-7.
        """
        super().__init__(learning_rate)
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
            # 1. Accumulate squared gradients: n_t = n_{t-1} + g_t^2
            self.accumulator[i][0] += grads[i][0] ** 2
            # 2. Calculate and apply update: θ_t = θ_{t-1} - η * g_t / sqrt(n_t + ε)
            params[i][0] -= self.lr * grads[i][0] / (np.sqrt(self.accumulator[i][0]) + self.epsilon)

            # --- Bias Updates ---
            # 1. Accumulate squared gradients
            self.accumulator[i][1] += grads[i][1] ** 2
            # 2. Calculate and apply update
            params[i][1] -= self.lr * grads[i][1] / (np.sqrt(self.accumulator[i][1]) + self.epsilon)