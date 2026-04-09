#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>
#include "../layers/Layer.h"
#include "../Loss/Loss.h"
#include "../optimizers/Optimizer.h"
#include "../metrics/Accuracy.h"

class Model {

protected:
    std::vector<Layer*> layers_;

    Eigen::MatrixXf Forward(const Eigen::MatrixXf &input) {
        Eigen::MatrixXf current_input = input;
        Eigen::MatrixXf current_output;

        for (Layer *layer : layers_) {
            layer->Forward(current_input, current_output);
            current_input = current_output;
        }
        return current_output;
    }
    void Backward(Loss &loss_function, const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) {

        Eigen::MatrixXf current_grad;
        Eigen::MatrixXf input_grad;

        loss_function.Backward(predictions, targets, current_grad);

        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            Layer *layer = *it;
            layer->Backward(input_grad, current_grad);
            current_grad = input_grad;
        }

    }

    void UpdateParams(Optimizer &optimizer) {
        for (Layer *layer :layers_) {
            layer->UpdateParams(&optimizer);
        }
    }

public:
    Model() = default;
    ~Model() = default;

    void AddLayer(Layer *layer) {
        layers_.push_back(layer);
    }

    void Fit(int epochs, int batch_size, const Eigen::MatrixXf &input, const Eigen::MatrixXf &labels, Loss &loss_function, Optimizer &optimizer) {

        int num_samples = input.rows();
        // Calcula a quantidade total de lotes (arredondando para cima se sobrar resto)
        int num_batches = (num_samples + batch_size - 1) / batch_size;

        std::ofstream historico("historico.csv");
        historico << "Epoch,Loss,Accuracy\n";

        // Loop 1: Épocas
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            float epoch_accuracy = 0.0f; // <-- Variável nova para somar a acurácia!

            // Loop 2: Lotes (Batches)
            for (int i = 0; i < num_samples; i += batch_size) {

                int current_batch = std::min(batch_size, num_samples - i);

                Eigen::MatrixXf X_batch = input.block(i, 0, current_batch, input.cols());
                Eigen::MatrixXf Y_batch = labels.block(i, 0, current_batch, labels.cols());

                // 1. Passe Direto (Forward)
                Eigen::MatrixXf predictions = Forward(X_batch);

                // 2. Calcula o Erro (Loss)
                float batch_loss = 0.0f;
                loss_function.Forward(predictions, Y_batch, batch_loss);
                epoch_loss += batch_loss;

                // 3. Calcula a Acurácia (Porcentagem de acerto do lote)
                // Se você já adicionou o calculateAccuracy no Model.h, chamamos ele aqui!
                epoch_accuracy += CalculateAccuracy(predictions, Y_batch);

                // 4. Passe Reverso (Backward)
                Backward(loss_function, predictions, Y_batch);

                // 5. Atualiza os Pesos (Optimizer)
                UpdateParams(optimizer);
            }

            // Calculamos a média da época
            float avg_loss = epoch_loss / num_batches;
            float avg_acc = (epoch_accuracy / num_batches) * 100.0f; // Multiplica por 100 para virar %

            // Agora sim! O print digno de um framework profissional:
            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << " - Loss: " << avg_loss
                      << " - Accuracy: " << avg_acc << "%" << std::endl;
                        historico << epoch+1 << "," << avg_loss << "," << avg_acc << "\n";

        }
    }

    void Evaluate(const Eigen::MatrixXf &input, const Eigen::MatrixXf &labels) {
                Eigen::MatrixXf predictions = Forward(input);
                float accuracy = CalculateAccuracy(predictions, labels);
                std::cout << "Test Accuracy: " << accuracy << std::endl;
    }

    void SaveModel(std::ofstream &file) {
        for (Layer *layer : layers_) {
            layer->SaveParams(file);
        }
    }

};