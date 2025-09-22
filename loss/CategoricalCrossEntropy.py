import numpy as np
from loss.base import LossFunction  # Supondo que sua classe base esteja aqui


class CategoricalCrossEntropy(LossFunction):
    """Implementa a função de perda de Entropia Cruzada Categórica.

    Esta é a função de perda padrão para problemas de classificação multi-classe.
    Ela mede a dissimilaridade entre a distribuição de probabilidade prevista
    pela rede (saída da Softmax) e a distribuição real (os rótulos one-hot).
    """

    def __init__(self):
        """Inicializa a função de perda."""
        super().__init__()

    def forward(self, y_pred, y_true):
        """Calcula a perda média de Entropia Cruzada Categórica para um lote.

        Args:
            y_pred (np.ndarray): As distribuições de probabilidade previstas pela
                rede, com shape (batch_size, num_classes).
            y_true (np.ndarray): Os rótulos verdadeiros em formato one-hot, com
                o mesmo shape de y_pred.

        Returns:
            float: O valor da perda média para o lote.
        """
        # Número de amostras no lote
        m = y_true.shape[0]
        # "Clipamos" as previsões para evitar o log(0), que resultaria em erro (nan)
        y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)

        # Calcula a perda total e a normaliza pelo tamanho do lote
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        return loss

    def backward(self, y_pred, y_true):
        """Calcula o gradiente inicial da perda para a retropropagação.

        NOTA: Esta é a derivada simplificada da combinação Softmax + Entropia Cruzada,
        o que torna o cálculo do gradiente inicial muito eficiente.

        Args:
            y_pred (np.ndarray): As distribuições de probabilidade previstas.
            y_true (np.ndarray): Os rótulos verdadeiros em formato one-hot.

        Returns:
            np.ndarray: O gradiente da perda em relação à saída da rede, pronto
                para ser passado para o backward da última camada.
        """
        # Normaliza o gradiente pelo tamanho do lote
        m = y_true.shape[0]
        grad_output = (y_pred - y_true) / m
        return grad_output