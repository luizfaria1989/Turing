from optimizers.base import Optimizer # Supondo que sua classe base esteja aqui

class GradientDescent(Optimizer):
    """Implementa o otimizador de Gradiente Descendente (GD).

    Esta classe herda da classe base Otimizador e implementa a lógica
    de atualização de parâmetros seguindo a fórmula clássica:
    θ_novo = θ_antigo - α * ∇J(θ)

    Attributes:
        lr (float): A taxa de aprendizado (α) usada para a atualização.
    """
    def __init__(self, learning_rate=0.001):
        """Inicializa o otimizador GradientDescent.

        Args:
            learning_rate (float, optional): A taxa de aprendizado (α).
                Padrão: 0.001.
        """
        super().__init__(learning_rate)

    def step(self, params, grads):
        """Executa um único passo de otimização para todos os parâmetros.

        Args:
            params (list): Uma lista de listas contendo os parâmetros do modelo.
                Estrutura esperada: [[pesos1, biases1], [pesos2, biases2], ...]
            grads (list): Uma lista de listas contendo os gradientes para cada
                parâmetro, com a mesma estrutura de `params`.
        """
        # Itera sobre cada grupo de parâmetros (um por camada Densa)
        for i in range(len(params)):
            # A fórmula: parametro -= learning_rate * gradiente_do_parametro
            params[i][0] -= self.lr * grads[i][0]  # Atualiza pesos (w)
            params[i][1] -= self.lr * grads[i][1]  # Atualiza biases (b)