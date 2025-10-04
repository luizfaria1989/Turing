import numpy as np
from optimizers.base import Optimizer


class Adam(Optimizer):
    """Implements the Adam (Adaptive Moment Estimation) optimizer.

    Adam combines the ideas of Momentum (using a moving average of the gradient)
    and RMSProp (using a moving average of the squared gradient). It computes
    adaptive learning rates for each parameter and includes a bias-correction
    step to account for the initialization of the moving averages at zero.

    The update formulas are:
    1. m_t = β1 * m_{t-1} + (1 - β1) * g_t
    2. v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
    3. m_hat = m_t / (1 - β1^t)
    4. v_hat = v_t / (1 - β2^t)
    5. θ_t = θ_{t-1} - η * m_hat / (sqrt(v_hat) + ε)

    Attributes:
        lr (float): The learning rate (η).
        beta_1 (float): The decay rate for the first moment estimate (m).
        beta_2 (float): The decay rate for the second moment estimate (v).
        epsilon (float): A small constant for numerical stability (ε).
        m (list): The 'memory' for the first moment (mean of gradients).
        v (list): The 'memory' for the second moment (uncentered variance).
        t (int): The timestep counter for bias correction.
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        """Initializes the Adam optimizer.

        Args:
            learning_rate (float, optional): The learning rate (η). Defaults to 0.001.
            beta_1 (float, optional): The decay rate for the first moment estimate.
                Defaults to 0.9.
            beta_2 (float, optional): The decay rate for the second moment estimate.
                Defaults to 0.999.
            epsilon (float, optional): A small value for numerical stability (ε).
                Defaults to 1e-7.
        """
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None  # First moment vector (like Momentum's velocity)
        self.v = None  # Second moment vector (like RMSProp's accumulator)
        self.t = 0  # Timestep counter, starts at 0

    def step(self, params, grads):
        """Performs a single optimization step."""
        self.t += 1

        if self.m is None:
            self.m = [[np.zeros_like(p) for p in group] for group in params]
        if self.v is None:
            self.v = [[np.zeros_like(p) for p in group] for group in params]

        for i in range(len(params)):
            # --- Weight Updates ---
            # Update biased first moment estimate: m_t = β1*m_{t-1} + (1-β1)*g_t
            self.m[i][0] = self.beta_1 * self.m[i][0] + (1 - self.beta_1) * grads[i][0]
            # Update biased second raw moment estimate: v_t = β2*v_{t-1} + (1-β2)*g_t^2
            self.v[i][0] = self.beta_2 * self.v[i][0] + (1 - self.beta_2) * (grads[i][0] ** 2)

            # Compute bias-corrected first moment estimate: m_hat = m_t / (1 - β1^t)
            m_hat = self.m[i][0] / (1 - self.beta_1 ** self.t)
            # Compute bias-corrected second raw moment estimate: v_hat = v_t / (1 - β2^t)
            v_hat = self.v[i][0] / (1 - self.beta_2 ** self.t)

            # Update parameters: θ_t = θ_{t-1} - η * m_hat / (sqrt(v_hat) + ε)
            params[i][0] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # --- Bias Updates ---
            self.m[i][1] = self.beta_1 * self.m[i][1] + (1 - self.beta_1) * grads[i][1]
            self.v[i][1] = self.beta_2 * self.v[i][1] + (1 - self.beta_2) * (grads[i][1] ** 2)

            m_hat_b = self.m[i][1] / (1 - self.beta_1 ** self.t)
            v_hat_b = self.v[i][1] / (1 - self.beta_2 ** self.t)

            params[i][1] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)