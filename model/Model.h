#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>
#include "../layers/Layer.h"
#include "../Loss/Loss.h"
#include "../optimizers/Optimizer.h"
#include "../metrics/Accuracy.h"

/**
 * @class Model
 * @brief Orquestrador principal da Rede Neural.
 * * A classe Model gerencia o grafo sequencial de camadas e orquestra o fluxo
 * de dados (Forward Pass), a retropropagação do erro (Backward Pass) e a
 * otimização de parâmetros. Ela transforma componentes isolados em um
 * sistema de aprendizado de máquina unificado.
 */
class Model {

protected:

    /// @brief Vetor ordenado de ponteiros para as camadas da rede.
    /// Define a topologia sequencial do modelo (Feedforward).
    std::vector<Layer*> layers_;

    /**
     * @brief Executa o passe direto (Forward Pass) em toda a rede.
     * * Transmite a matriz de entrada através de todas as camadas sequencialmente.
     * Matematicamente, representa a composição de funções da rede neural:
     * \f[\hat{Y} = f_L(f_{L-1}(\dots f_1(X) \dots))\f]
     * * @param input Matriz de entrada do lote atual \f$X\f$.
     * @return Matriz de previsão final \f$\hat{Y}\f$ gerada pela última camada.
     */
    Eigen::MatrixXf Forward(const Eigen::MatrixXf &input) {
        Eigen::MatrixXf current_input = input;
        Eigen::MatrixXf current_output;

        for (Layer *layer : layers_) {
            layer->Forward(current_input, current_output);
            current_input = current_output;
        }
        return current_output;
    }

    /**
     * @brief Executa o passe reverso (Backward Pass / Backpropagation).
     * * Inicia o cálculo do gradiente utilizando a função de perda e, em seguida,
     * propaga o erro de trás para frente aplicando a Regra da Cadeia sequencialmente.
     * Para cada camada $l$, a propagação do gradiente $\delta$ segue a relação:
     * \f[\delta^{(l-1)} = \text{Layer}^{(l)}\text{.Backward}(\delta^{(l)})\f]
     * * @param loss_function Referência para a função de perda instanciada.
     * @param predictions Matriz de previsões \f$\hat{Y}\f$ gerada no Forward Pass.
     * @param targets Matriz de rótulos reais \f$Y\f$.
     */
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

    /**
     * @brief Delega a atualização de parâmetros para todas as camadas.
     * @param optimizer Referência para o algoritmo de otimização escolhido.
     */
    void UpdateParams(Optimizer &optimizer) {
        for (Layer *layer :layers_) {
            layer->UpdateParams(&optimizer);
        }
    }

public:

    /**
     * @brief Construtor padrão da classe Model.
     */
    Model() = default;

    /**
     * @brief Destrutor padrão.
     * @note Como os ponteiros das camadas são fornecidos via AddLayer, a
     * responsabilidade de liberar a memória das camadas recai sobre quem as instanciou.
     */
    ~Model() = default;

    /**
     * @brief Adiciona uma nova camada sequencial ao modelo.
     * @param layer Ponteiro para a camada que será empilhada na arquitetura.
     */
    void AddLayer(Layer *layer) {
        layers_.push_back(layer);
    }

    /**
     * @brief Realiza o treinamento do modelo (Mini-Batch Gradient Descent).
     * * Este é o loop principal de treinamento. Ele subdivide os dados em minilotes (batches),
     * executa as etapas de Feedforward, Loss, Backpropagation e Otimização iterativamente.
     * Além disso, registra a evolução das métricas em um arquivo CSV para análise posterior.
     * * @param epochs Número de vezes que o modelo iterará sobre todo o conjunto de dados.
     * @param batch_size Quantidade de amostras processadas simultaneamente antes de uma atualização de pesos.
     * @param input Matriz contendo o conjunto completo de dados de treinamento.
     * @param labels Matriz contendo os rótulos reais do conjunto de treinamento.
     * @param loss_function Referência para a métrica de erro objetivo.
     * @param optimizer Referência para a regra de atualização dos pesos.
     */
    void Fit(int epochs, int batch_size, const Eigen::MatrixXf &input, const Eigen::MatrixXf &labels, Loss &loss_function, Optimizer &optimizer) {

        std::cout << "Iniciando o treinamento por " << epochs << " epocas..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        int num_samples = input.rows();
        int num_batches = (num_samples + batch_size - 1) / batch_size;

        // Logger para gerar gráficos da curva de aprendizado posteriormente
        std::ofstream historico("historico.csv");
        historico << "Epoch,Loss,Accuracy\n";

        // Loop de Épocas
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            float epoch_accuracy = 0.0f;

            // Loop de Mini-Lotes (Batches)
            for (int i = 0; i < num_samples; i += batch_size) {

                int current_batch = std::min(batch_size, num_samples - i);

                // Fatiamento (Slicing) dos dados originais
                Eigen::MatrixXf X_batch = input.block(i, 0, current_batch, input.cols());
                Eigen::MatrixXf Y_batch = labels.block(i, 0, current_batch, labels.cols());

                // 1. Passe Direto (Forward)
                Eigen::MatrixXf predictions = Forward(X_batch);

                // 2. Avaliação de Erro
                float batch_loss = 0.0f;
                loss_function.Forward(predictions, Y_batch, batch_loss);
                epoch_loss += batch_loss;

                // 3. Avaliação de Métrica
                epoch_accuracy += CalculateAccuracy(predictions, Y_batch);

                // 4. Passe Reverso (Retropropagação)
                Backward(loss_function, predictions, Y_batch);

                // 5. Otimização (Atualização de Pesos)
                UpdateParams(optimizer);
            }

            // Médias agregadas da época atual
            float avg_loss = epoch_loss / num_batches;
            float avg_acc = (epoch_accuracy / num_batches) * 100.0f;

            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << " - Loss: " << avg_loss
                      << " - Accuracy: " << avg_acc << "%" << std::endl;
                        historico << epoch+1 << "," << avg_loss << "," << avg_acc << "\n";

        }
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Treinamento concluído com sucesso! \0/" << std::endl;
    }

    /**
     * @brief Avalia o desempenho do modelo treinado em um conjunto de testes isolado.
     * * Processa os dados em Modo de Inferência (sem chamar o backward pass ou atualizar pesos).
     * * @param input Matriz de dados de teste (dados invisíveis durante o treinamento).
     * @param labels Rótulos reais dos dados de teste.
     */
    void Evaluate(const Eigen::MatrixXf &input, const Eigen::MatrixXf &labels) {
                Eigen::MatrixXf predictions = Forward(input);
                float accuracy = CalculateAccuracy(predictions, labels);
                std::cout << "Test Accuracy: " << accuracy << std::endl;
    }

    /**
     * @brief Persiste o estado interno (pesos e vieses treinados) de todo o modelo em disco.
     * @param file Referência para o fluxo de saída em arquivo.
     */
    void SaveModel(std::ofstream &file) {
        for (Layer *layer : layers_) {
            layer->SaveParams(file);
        }
    }

};