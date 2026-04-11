#pragma once
#include "Optimizer.h"

/**
 * @class GradientDescent
 * @brief Implementa o algoritmo de otimização Gradiente Descendente (Standard Gradient Descent).
 * * É o método fundamental de otimização de primeira ordem para redes neurais.
 * Ele atualiza os parâmetros do modelo iterativamente na direção oposta ao gradiente
 * da função de perda, buscando o mínimo da função objetivo.
 * A regra matemática de transição de estado para um parâmetro genérico \f$ \theta \f$ é definida como:
 * \f[ \theta_{t+1} = \theta_t - \eta \nabla_{\theta} L \f]
 * Onde \f$ \eta \f$ representa a taxa de aprendizado (learning rate).
 */
class GradientDescent : public Optimizer{

public:

    /**
    * @brief Construtor do otimizador Gradiente Descendente.
    * @param learning_rate O tamanho do passo \f$ \eta \f$ dado na direção do gradiente negativo. O valor padrão é 0.01.
    */
    GradientDescent(float learning_rate = 0.01f) : Optimizer(learning_rate) {}

    /**
     * @brief Destrutor padrão.
     */
    virtual ~GradientDescent() = default;

    /**
    * @brief Executa a atualização dos pesos e vieses da camada.
    * * Aplica a regra algébrica clássica do Gradiente Descendente diretamente
    * subtraindo o gradiente escalonado das matrizes de parâmetros originais:
    * \f[ W = W - \eta \frac{\partial L}{\partial W} \f]
    * \f[ b = b - \eta \frac{\partial L}{\partial b} \f]
    * * @param weights Referência para a matriz de pesos \f$ W \f$ da camada, que será atualizada in-place.
    * @param biases Referência para o vetor de vieses \f$ b \f$ da camada, que será atualizado in-place.
    * @param grad_weights Matriz constante contendo os gradientes da perda em relação aos pesos \f$ \frac{\partial L}{\partial W} \f$.
    * @param grad_biases Vetor constante contendo os gradientes da perda em relação aos vieses \f$ \frac{\partial L}{\partial b} \f$.
    */
    void Update (Eigen::MatrixXf &weights, Eigen::MatrixXf &biases, const Eigen::MatrixXf &grad_weights, const Eigen::MatrixXf &grad_biases) override {
        weights = weights - (learning_rate_ * grad_weights);
        biases = biases - (learning_rate_ * grad_biases);
    };

};