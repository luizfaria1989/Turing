import numpy as np

from tqdm import tqdm
import time # Apenas para o exemplo, para ver a barra se mover

from layers.Convolution2D import Convolution2D
from layers.Dense import Dense
from layers.MaxPooling2D import MaxPooling2D
from layers.activation.Tanh import Tanh
from layers.Flatten import Flatten
from layers.activation.Softmax import Softmax
from loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from optimizers.GradientDescent import GradientDescent
from model.Sequential import Sequential
from datasets.mnist import load_mnist
from metrics.Accuracy import Accuracy



# =====================
# Carregar e preparar dados
# =====================
x_train, y_train, x_test, y_test = load_mnist()

# Normaliza as imagens
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Converter para int64
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

# One-hot encoding
num_classes = 10
y_train_one_hot = np.zeros((y_train.size, num_classes))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

y_test_one_hot = np.zeros((y_test.size, num_classes))
y_test_one_hot[np.arange(y_test.size), y_test] = 1

# =====================
# Modelo
# =====================
model = Sequential([
    Convolution2D(input_channels=1, num_filters=16, kernel_size=(3, 3), padding=1),
    Tanh(),
    MaxPooling2D(pool_size=(2, 2)),

    Convolution2D(input_channels=16, num_filters=32, kernel_size=(3, 3), padding=1),
    Tanh(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(input_size=7 * 7 * 32, output_size=128),
    Tanh(),
    Dense(input_size=128, output_size=10),
    Softmax()
])

loss_func = CategoricalCrossEntropy()
optimizer_method = GradientDescent(learning_rate=0.001)


# =====================
# Função de treino
# =====================
def train(epochs=100, batch_size=32):
    n_samples = x_train.shape[0]

    # O loop das épocas continua o mesmo
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0  # Vamos calcular a acurácia do treino também

        # ... (código de shuffle dos dados) ...
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train_one_hot[indices]

        # <-- 1. CRIE A BARRA DE PROGRESSO AQUI
        with tqdm(range(0, n_samples, batch_size), desc=f"Época {epoch + 1}/{epochs}") as pbar:
            # O loop agora usa a barra de progresso 'pbar'
            for start in pbar:
                end = min(start + batch_size, n_samples)
                x_batch = x_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # Forward
                preds = model.forward(x_batch)

                # Cálculo da perda
                loss = loss_func.forward(preds, y_batch)
                epoch_loss += loss

                # Gradiente inicial
                initial_gradient = loss_func.backward(preds, y_batch)

                # Backward
                grads = model.backward(initial_gradient)

                # Atualização dos pesos
                optimizer_method.step(model.params, grads)

                # <-- 2. ATUALIZE A BARRA COM MÉTRICAS (OPCIONAL, MAS MUITO ÚTIL)
                pbar.set_postfix(loss=f"{loss:.4f}")

        # ... (seu código de avaliação no final da época) ...
        test_preds = model.forward(x_test)
        test_loss = loss_func.forward(test_preds, y_test_one_hot)
        accuracy = Accuracy()
        test_accuracy = accuracy.calculate(test_preds, y_test_one_hot)

        print(
            f'Fim da Época {epoch + 1}/{epochs} | Perda no teste: {test_loss:.4f} | Acurácia no teste: {test_accuracy:.4f}')


# =====================
# Execução
# =====================
if __name__ == '__main__':
    train()
