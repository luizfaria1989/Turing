#include "../model/Model.h"
#include "../layers/DenseLayer.h"
#include "../layers/Tanh.h"
#include "../layers/Softmax.h"
#include "../loss/CategoricalCrossEntropy.h"
#include "../optimizers/GradientDescent.h"

int main() {
    Eigen::MatrixXf X_train(4, 2);
    X_train << 0, 0,
               0, 1,
               1, 0,
               1, 1;

    Eigen::MatrixXf Y_train(4, 2);
    Y_train << 1, 0,  // 0 XOR 0 = Falso
               0, 1,  // 0 XOR 1 = Verdadeiro
               0, 1,  // 1 XOR 0 = Verdadeiro
               1, 0;  // 1 XOR 1 = Falso

    Model model;

    model.AddLayer(new DenseLayer(2, 4));
    model.AddLayer(new Tanh());

    model.AddLayer(new DenseLayer(4, 2));
    model.AddLayer(new Softmax());

    CategoricalCrossEntropy loss_function;

    GradientDescent optimizer(0.1f);

    int epochs = 2000;
    int batch_size = 4;

    std::cout << "Iniciando o treinamento do problema XOR...\n" << std::endl;
    model.Fit(epochs, batch_size, X_train, Y_train, loss_function, optimizer);

    std::cout << "\n------------------------------------------------" << std::endl;
    std::cout << "Treinamento concluído!" << std::endl;
    std::cout << "Avaliando a tabela verdade final:" << std::endl;

    model.Evaluate(X_train, Y_train);

    return 0;
}