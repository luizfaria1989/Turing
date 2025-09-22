from ..base import Layer

class ActivationFunction(Layer):
    """
    Classe base para as funções de ativação.
    Herda da classe Camada.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError