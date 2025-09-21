from Optimizers.base import Optimizer

class GradientDescent(Optimizer):
    """Implementa o otimizador de Gradiente Descendente (GD).

    Esta classe herda da classe base Otimizador e implementa a lógica
    de atualização de parâmetros seguindo a fórmula: θ = θ - α * ∇J(θ).

    Attributes:
        lr (float): A taxa de aprendizado (α) usada para a atualização.
    """
    def __init__(self, learning_rate=0.001):
        """Inicializa o otimizador GradientDescent.

            Args: learning_rate (float, optional): A taxa de aprendizado (α). Padrão: 0.01.
        """
        super().__init__(learning_rate)

    def step(self, parameters, gradient):
        """Executa um único passo de otimização do gradiente descendente.

            Modifica os parâmetros "in-place", ou seja, altera diretamente a lista
            de parâmetros fornecida.

            Args:
                parameters (list of np.ndarray): Uma lista de arrays (numpy) contendo
                                                     os parâmetros do modelo a serem atualizados.
                gradients (list of np.ndarray): Uma lista de arrays (numpy) com os
                                                    gradientes correspondentes a cada parâmetro.

            Returns:
                None
        """
        for param, grad in zip(parameters, gradient):
            param -= self.lr * grad
