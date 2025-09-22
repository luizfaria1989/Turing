import numpy as np
from layers.base import Layer  # Supondo que sua classe base esteja aqui


class Dense(Layer):
    """Implementa uma camada densa (fully connected) da rede neural.

    Uma camada densa realiza uma transformação linear nos dados de entrada
    (Y = XW + b). É o principal bloco de construção para redes neurais.

    Attributes:
        w (np.ndarray): A matriz de pesos da camada.
        b (np.ndarray): O vetor de biases da camada.
        params (list): Uma lista contendo os parâmetros treináveis da camada, [w, b].
        activation (Layer, optional): A camada de ativação a ser aplicada após a
            transformação linear.
    """

    def __init__(self, input_size, output_size, activation=None):
        """Inicializa a camada Densa.

        Args:
            input_size (int): O número de features de entrada (neurônios da camada anterior).
            output_size (int): O número de neurônios de saída desta camada.
            activation (Layer, optional): Uma camada de ativação a ser usada.
                Padrão é None (nenhuma ativação).
        """
        super().__init__()
        self.w = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.input = None
        self.params = [self.w, self.b]
        self.activation = activation

    def forward(self, input_data):
        """Executa a passagem para frente (forward pass) da camada.

        Args:
            input_data (np.ndarray): Os dados de entrada com shape (batch_size, input_size).

        Returns:
            np.ndarray: A saída da camada com shape (batch_size, output_size).
        """
        self.input = input_data
        # Cálculo da transformação linear
        linear_output = self.input @ self.w + self.b

        # Aplica a função de ativação, se houver
        if self.activation:
            return self.activation.forward(linear_output)

        return linear_output

    def backward(self, output_gradient):
        """Executa a passagem para trás (backward pass) da camada.

        Calcula os gradientes para os pesos, biases e para a entrada da camada.

        Args:
            output_gradient (np.ndarray): O gradiente da perda em relação à saída desta camada.

        Returns:
            tuple[np.ndarray, list]: Uma tupla contendo:
                - O gradiente em relação à entrada da camada (para retropropagar).
                - Uma lista com os gradientes dos pesos e biases.
        """
        # Se houver uma função de ativação, primeiro fazemos o backward dela
        if self.activation:
            output_gradient = self.activation.backward(output_gradient)

        # Gradientes para os parâmetros
        grad_w = self.input.T @ output_gradient
        grad_b = np.sum(output_gradient, axis=0, keepdims=True)

        # Gradiente para passar para a camada anterior
        input_gradient = output_gradient @ self.w.T

        return input_gradient, [grad_w, grad_b]