#include <iostream>
#include <string>

// Inclua todos os seus cabeçalhos aqui (ajuste os caminhos conforme suas pastas)
#include "model/Model.h"
#include "layers/DenseLayer.h"
#include "layers/ReLU.h"
#include "layers/Softmax.h"
#include "loss/CategoricalCrossEntropy.h"
#include "optimizers/GradientDescent.h"
#include "MNISTLoader.h" // Onde você salvou o código da mensagem anterior
#include "optimizers/GDMomentum.h"
#include "optimizers/NAG.h"

int main() {
    std::cout << "Iniciando o framework de Deep Learning em C++..." << std::endl;

    // 1. Carregando os Dados
    // IMPORTANTE: Ajuste este caminho para a pasta onde você extraiu os arquivos do MNIST
    std::string path_images = "../data/train-images-idx3-ubyte/train-images-idx3-ubyte";
    std::string path_labels = "../data/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    std::string path_images_test = "../data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    std:: string path_labels_test = "../data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";

    std::cout << "Carregando o dataset MNIST..." << std::endl;
    Eigen::MatrixXf X_train, Y_train;

    try {
        X_train = MNISTLoader::loadImages(path_images);
        Y_train = MNISTLoader::loadLabels(path_labels);
    } catch (const std::exception& e) {
        std::cerr << "ERRO FATAL: " << e.what() << std::endl;
        std::cerr << "Verifique se os arquivos do MNIST estão na pasta correta e descompactados!" << std::endl;
        return 1;
    }

    std::cout << "Imagens carregadas: " << X_train.rows() << " x " << X_train.cols() << std::endl;
    std::cout << "Labels carregadas:  " << Y_train.rows() << " x " << Y_train.cols() << std::endl;

    // 2. Montando a Arquitetura da Rede Neural
    std::cout << "Montando a rede neural..." << std::endl;
    Model model;

    // Camada Oculta: 784 entradas (pixels 28x28) para 128 neurônios
    model.AddLayer(new DenseLayer(784, 128));
    model.AddLayer(new ReLU());

    // Camada de Saída: 128 entradas para 10 neurônios (classes de 0 a 9)
    model.AddLayer(new DenseLayer(128, 10));
    model.AddLayer(new Softmax());

    // 3. Configurando a Bússola e o Motor
    CategoricalCrossEntropy loss_function;

    // Usamos uma taxa de aprendizado (learning rate) de 0.1, que funciona bem para MNIST com SGD
    NAG optimizer(0.01f, 0.01f);

    // 4. O Treinamento (A Mágica Acontece Aqui!)
    int epochs = 20;
    int batch_size = 64;

    // Chama o método fit que nós construímos!
    model.Fit(epochs, batch_size, X_train, Y_train, loss_function, optimizer);

    std::cout << "Carregando o dataset MNIST..." << std::endl;
    Eigen::MatrixXf X_test, Y_test;

    try {
        X_test = MNISTLoader::loadImages(path_images_test);
        Y_test = MNISTLoader::loadLabels(path_labels_test);
    } catch (const std::exception& e) {
        std::cerr << "ERRO FATAL: " << e.what() << std::endl;
        std::cerr << "Verifique se os arquivos do MNIST estão na pasta correta e descompactados!" << std::endl;
        return 1;
    }

    model.Evaluate(X_test, Y_test);

    // (Opcional) Limpeza de memória: num projeto real, o destrutor do Model
    // deveria dar um 'delete' em cada ponteiro do std::vector para evitar memory leaks.

    return 0;
}
    // TIP See CLion help at <a href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>. Also, you can try interactive lessons for CLion by selecting 'Help | Learn IDE Features' from the main menu