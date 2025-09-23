import numpy as np
from layers.Dense import Dense
from layers.activation.ReLU import ReLU
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

# Flatten das imagens
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

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
    Dense(784, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
])

loss_func = CategoricalCrossEntropy()
optimizer_method = GradientDescent(learning_rate=0.001)

# =====================
# Função de treino
# =====================
def train(epochs=100, batch_size=32):
    n_samples = x_train.shape[0]

    for epoch in range(epochs):
        # ... (código de shuffle dos dados, sem alterações) ...
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train_one_hot[indices]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Forward
            preds = model.forward(x_batch)

            # Cálculo da perda
            loss = loss_func.forward(preds, y_batch)

            # Gradiente inicial
            initial_gradient = loss_func.backward(preds, y_batch)

            # Backward - agora só calcula e retorna os gradientes
            grads = model.backward(initial_gradient)

            # Atualização dos pesos - agora é feita pelo otimizador!
            optimizer_method.step(model.params, grads)

        # ... (código de avaliação, sem alterações) ...
        test_preds = model.forward(x_test)
        test_loss = loss_func.forward(test_preds, y_test_one_hot)
        # Supondo que sua classe Accuracy tenha um método estático ou de classe
        accuracy = Accuracy()
        test_accuracy = accuracy.calculate(test_preds, y_test_one_hot)

        print(f'Época {epoch+1}/{epochs} | Perda no teste: {test_loss} | Acurácia no teste: {test_accuracy:.4f}')


# =====================
# Execução
# =====================
if __name__ == '__main__':
    train()
