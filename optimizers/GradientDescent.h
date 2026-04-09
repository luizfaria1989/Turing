#pragma once
#include "Optimizer.h"

/**
 * @brief Implementa um otimizador baseado em gradiente descendente.
 * * Responsável por calcular a atualização dos pesos e vieses do modelo treinado.
 * @note A fórmula matemática utilizada é X_(t+1) = X_(t) - alpha * grad
 */
class GradientDescent : public Optimizer{

public:

    /**
     * @brief Construtor padrão da otimizador gradiente descente.
     * @note Por padrão, o tamanho do passo é dado por 0.01.
     */
    GradientDescent(float learning_rate = 0.01f) : Optimizer(learning_rate) {}

    /**
     * @brief Destrutor padrão.
     */
    virtual ~GradientDescent() = default;

    /**
    * @brief Implementa o método de atualização de parâmetros do otimizador gradiente descendente.
    * @param weights Referência para a matriz de pesos da camada analisada.
    * @param biases Referência para o vetor de vieses da camada analisada.
    * @param grad_weights Referência constante para a matriz que armazena o gradiente dos pesos da camada analisada.
    * @param grad_biases Referência constante para o vetor que armazena o gradiente dos vieses da camanda analisada.
    * @note A fórmula matemática executada é: W = W - alpha * dW para os pesos, e B = B - alpha * dB.
    */
    void Update (Eigen::MatrixXf &weights, Eigen::MatrixXf &biases, const Eigen::MatrixXf &grad_weights, const Eigen::MatrixXf &grad_biases) override {
        weights = weights - (learning_rate_ * grad_weights);
        biases = biases - (learning_rate_ * grad_biases);
    };

};