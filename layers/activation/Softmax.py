import numpy as np
from layers.base import Layer # Supondo que sua classe base esteja aqui

class Softmax(Layer):
    """Implementa a camada de ativação Softmax.

    A Softmax é tipicamente usada como a camada de ativação final em problemas
    de classificação multi-classe. Ela transforma um vetor de scores (logits)
    em uma distribuição de probabilidades, onde cada valor está no intervalo
    [0, 1] e a soma de todos os valores é 1.

    Attributes:
        probs (np.ndarray): Armazena as probabilidades de saída do forward pass.
    """
    def __init__(self):
        """Inicializa a camada Softmax."""
        super().__init__()
        self.probs = None

    def forward(self, x):
        """Executa a passagem para frente (forward pass) da camada.

        Args:
            x (np.ndarray): Os dados de entrada (logits) da camada anterior, com
                shape (batch_size, num_classes).

        Returns:
            np.ndarray: As probabilidades de saída, com o mesmo shape da entrada.
        """
        # A subtração pelo máximo é um truque para estabilidade numérica,
        # prevenindo overflow com valores de entrada muito grandes.
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.probs

    def backward(self, grad_output):
        """Executa a passagem para trás (backward pass) da camada.

        NOTA IMPORTANTE: Este é um backward pass simplificado que SÓ funciona
        quando a Softmax é usada em conjunto com a função de perda de
        Entropia Cruzada Categórica. A derivada complexa da Softmax é
        convenientemente cancelada pela derivada da Entropia Cruzada,
        resultando em uma simples passagem do gradiente.

        Args:
            grad_output (np.ndarray): O gradiente da perda em relação à saída
                desta camada. Neste design, ele será (y_previsto - y_real).

        Returns:
            tuple[np.ndarray, None]: Uma tupla contendo:
                - O gradiente em relação à entrada da camada.
                - None, pois a Softmax não possui parâmetros treináveis.
        """
        # O gradiente simplesmente passa direto pela camada.
        return grad_output, None