#pragma once
#include <Eigen/Dense>

/**
 * @brief Calcula a métrica de acurácia de classificação do modelo.
 * * Esta função avalia a taxa de acerto global das previsões em relação aos rótulos reais.
 * Ela é projetada para problemas de classificação (como o reconhecimento de dígitos MNIST),
 * onde a classe prevista é aquela com a maior probabilidade (ou logit) na camada de saída.
 * * Matematicamente, a acurácia para um lote de \f$ N \f$ amostras é calculada extraindo-se o
 * argumento máximo (\f$ \text{argmax} \f$) de cada linha e utilizando a Função Indicadora \f$ \mathbb{I} \f$:
 * \f[ \hat{c}_i = \arg\max_j (\hat{Y}_{i,j}) \f]
 * \f[ c_i = \arg\max_j (Y_{i,j}) \f]
 * \f[ \text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{c}_i = c_i) \f]
 * Onde \f$ \mathbb{I} \f$ retorna 1 se a classe prevista for igual à classe real, e 0 caso contrário.
 * * @param predictions Matriz constante \f$ \hat{Y} \f$ contendo as saídas do modelo (ex: probabilidades pós-Softmax).
 * @param targets Matriz constante \f$ Y \f$ contendo os rótulos reais (geralmente codificados em One-Hot Encoding).
 * @return Um valor escalar do tipo float no intervalo \f$ [0, 1] \f$ representando a proporção de acertos do lote.
 * * @note Na biblioteca Eigen, a operação vetorial equivalente ao \f$ \text{argmax} \f$ é
 * obtida através do método `maxCoeff()`, passando o endereço de um `Eigen::Index` para capturar a posição do maior valor.
 */
float CalculateAccuracy(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) {
    int correct_predictions = 0;
    int num_samples = predictions.rows(); // N = tamanho do batch

    for (int i = 0; i < num_samples; ++i) {
        Eigen::Index pred_index;
        Eigen::Index target_index;

        // Extrai o argmax da previsão e do alvo
        predictions.row(i).maxCoeff(&pred_index);
        targets.row(i).maxCoeff(&target_index);

        // Função indicadora lógica: soma 1 se houver match
        if (pred_index == target_index) {
            correct_predictions++;
        }
    }

    // Retorna a média de acertos
    return static_cast<float>(correct_predictions) / num_samples;
};