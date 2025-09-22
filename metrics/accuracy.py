import numpy as np


class Accuracy:
    """
    Calcula a métrica de acurácia para problemas de classificação.

    A acurácia mede a proporção de previsões corretas em relação ao
    número total de amostras. É calculada como:
    Acurácia = (Previsões Corretas) / (Total de Previsões)
    """

    def __init__(self):
        """Inicializa o calculador de acurácia."""
        pass

    def calculate(self, y_pred, y_true):
        """Calcula a acurácia para um lote de previsões.

        Args:
            y_pred (np.ndarray): As saídas previstas pelo modelo, geralmente
                como distribuições de probabilidade (saída da Softmax).
                Shape: (batch_size, num_classes).
            y_true (np.ndarray): Os rótulos verdadeiros em formato one-hot.
                Shape: (batch_size, num_classes).

        Returns:
            float: O valor da acurácia, um número entre 0 e 1.
        """
        # Converte as probabilidades em previsões de classe (o índice de maior valor)
        predictions = np.argmax(y_pred, axis=1)

        # Converte os rótulos one-hot de volta para classes
        correct_predictions = np.argmax(y_true, axis=1)

        # Compara as previsões com os valores reais e soma os acertos
        correct = np.sum(predictions == correct_predictions)

        # Calcula a acurácia
        accuracy = correct / len(predictions)

        return accuracy