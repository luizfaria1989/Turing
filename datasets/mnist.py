import os
import urllib.request
import numpy as np

def load_mnist():

    os.makedirs("datasets/_data", exist_ok=True)

    if not os.path.exists("datasets/_data/mnist.npz"):
        urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "datasets/_data/mnist.npz")

    with np.load("datasets/_data/mnist.npz") as mnist:
        x_train, y_train = mnist["x_train"], mnist["y_train"]
        x_test, y_test = mnist["x_test"], mnist["y_test"]

    return x_train, y_train, x_test, y_test
