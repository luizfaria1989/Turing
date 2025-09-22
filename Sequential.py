import numpy as np
# from layers.base import Layer # Supondo que você tenha uma classe base

class Sequential:
    """Um container para empilhar camadas de rede neural em uma sequência.

    Esta classe gerencia uma lista de camadas e orquestra a passagem de dados
    para frente (forward) e a retropropagação de gradientes para trás (backward).

    Attributes:
        layers (list): A lista de objetos de camada que compõem a rede.
        params (list): Uma lista de listas contendo os parâmetros treináveis
            (pesos e biases) de todas as camadas na rede que os possuem.
        grads (list): Uma lista de listas contendo os gradientes para cada
            parâmetro. Esta lista é populada após cada chamada ao `backward`.
    """
    def __init__(self, layers):
        """Inicializa o modelo Sequencial.

        Args:
            layers (list): Uma lista de objetos de camada (ex: [Dense(), ReLU()])
                na ordem em que devem ser aplicados.
        """
        self.layers = layers
        self.params = []
        self.grads = []

        # Loop inteligente que só coleta parâmetros de camadas que realmente os têm
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                self.params.append(layer.params)

    def forward(self, input_data):
        """Executa a passagem para frente (forward pass) para toda a rede.

        Args:
            input_data (np.ndarray): Os dados de entrada para a primeira camada.

        Returns:
            np.ndarray: A saída final da rede após passar por todas as camadas.
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, gradiente_saida):
        """Executa a passagem para trás (backward pass) para toda a rede.

        Args:
            gradiente_saida (np.ndarray): O gradiente inicial da função de perda
                em relação à saída da rede.

        Returns:
            list: Uma lista de listas contendo os gradientes para todos os
                parâmetros treináveis da rede.
        """
        self.grads = []
        grad = gradiente_saida

        # Itera sobre as camadas na ordem inversa
        for layer in reversed(self.layers):
            grad, layer_grads = layer.backward(grad)

            # Coleta os gradientes apenas de camadas que os retornaram
            if layer_grads:
                self.grads.insert(0, layer_grads)

        return self.grads