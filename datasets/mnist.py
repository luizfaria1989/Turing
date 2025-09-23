import os
import urllib.request
import numpy as np


def load_mnist():
    """Baixa o dataset MNIST (se não existir localmente) e o carrega.

    A função verifica se o arquivo `mnist.npz` existe na pasta `datasets/_data/`.
    Se não existir, ele é baixado da fonte de datasets do Keras/TensorFlow.
    Em seguida, os dados são carregados em arrays NumPy.

    Returns:
        tuple: Uma tupla contendo quatro arrays NumPy na seguinte ordem:
        (x_train, y_train, x_test, y_test).
        - x_train (np.ndarray): Imagens de treino com shape (60000, 28, 28).
        - y_train (np.ndarray): Rótulos de treino com shape (60000,).
        - x_test (np.ndarray): Imagens de teste com shape (10000, 28, 28).
        - y_test (np.ndarray): Rótulos de teste com shape (10000,).
    """
    # Garante que o diretório de dados exista
    os.makedirs("datasets/_data", exist_ok=True)

    data_path = "datasets/_data/mnist.npz"

    # Baixa o arquivo apenas se ele não existir
    if not os.path.exists(data_path):
        print("Baixando o dataset MNIST...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            data_path
        )
        print("Download completo.")

    # Carrega os dados do arquivo .npz
    with np.load(data_path) as mnist:
        x_train, y_train = mnist["x_train"], mnist["y_train"]
        x_test, y_test = mnist["x_test"], mnist["y_test"]

    return x_train, y_train, x_test, y_test