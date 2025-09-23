import os
import urllib.request
import numpy as np


def load_mnist():
    """ This function downloads the MNIST dataset (if it doesn't exist yet) and then loads it.
    The function starts by verifying that the MNIST dataset is downloaded by checking the file `mnist.npz` in
    the directory `datasets/_data`.
    If it doesn't exist yet, it downloads the dataset from Keras/TensorFlow.
    And then loads the dataset into memory in a numpy array.

    Returns:
        tuple: A tuple containing four numpy arrays in the following order:
        (x_train, y_train, x_test, y_test).
        - x_train (np.ndarray): Train images with dimensions (60000, 28, 28).
        - y_train (np.ndarray): Train labels with dimensions (60000,).
        - x_test (np.ndarray): Test images with dimensions (10000, 28, 28).
        - y_test (np.ndarray): Test labels with dimensions (10000,).
    """
    # Makes sure that `datasets/_data` exists
    os.makedirs("datasets/_data", exist_ok=True)

    data_path = "datasets/_data/mnist.npz"

    # Download the file if it doesn`t exist yet
    if not os.path.exists(data_path):
        print("Dowloading MNIST dataset...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            data_path
        )
        print("Downloading complete.")

    # Load the data in the file .npz
    with np.load(data_path) as mnist:
        x_train, y_train = mnist["x_train"], mnist["y_train"]
        x_test, y_test = mnist["x_test"], mnist["y_test"]

    return x_train, y_train, x_test, y_test