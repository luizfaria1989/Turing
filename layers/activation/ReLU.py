import numpy as np
from layers.base import Layer  # Supondo que sua classe base esteja aqui


class ReLU(Layer):
    """Implementa a camada de ativação ReLU (Rectified Linear Unit).

    A ReLU aplica a função não-linear f(x) = max(0, x) elemento a elemento.
    É uma das funções de ativação mais comuns para camadas ocultas em redes
    neurais por ser computacionalmente eficiente e ajudar a mitigar o problema
    do desaparecimento do gradiente.

    Attributes:
        input (np.ndarray): Armazena a entrada do forward pass para ser usada
            no cálculo do gradiente durante o backward pass.
    """

    def __init__(self):
        """Inicializa a camada ReLU."""
        super().__init__()
        self.input = None

    def forward(self, input_data):
        """Executa a passagem para frente (forward pass) da camada.

        Args:
            input_data (np.ndarray): Os dados de entrada da camada anterior.

        Returns:
            np.ndarray: A saída da camada após a aplicação da ReLU, com o
                mesmo shape da entrada.
        """
        # Guarda a entrada para o backward pass
        self.input = input_data
        return np.maximum(0, self.input)

    def backward(self, grad_output):
        """Executa a passagem para trás (backward pass) da camada.

        Calcula o gradiente da perda em relação à entrada da camada. A derivada
        da ReLU é 1 para entradas > 0 e 0 caso contrário.

        Args:
            grad_output (np.ndarray): O gradiente da perda em relação à saída
                desta camada.

        Returns:
            tuple[np.ndarray, None]: Uma tupla contendo:
                - O gradiente em relação à entrada da camada (para retropropagar).
                - None, pois a ReLU não possui parâmetros treináveis.
        """
        # Cria uma máscara booleana e a multiplica pelo gradiente de saída
        grad_input = grad_output * (self.input > 0)

        return grad_input, None