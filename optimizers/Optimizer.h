#pragma once
#include <Eigen/Dense>

/**
 * @class Optimizer
 * @brief Classe base abstrata para todos os algoritmos de otimização.
 * * Define a interface para os métodos que atualizam os parâmetros treináveis
 * (pesos e vieses) da rede neural com o objetivo de minimizar a função de perda \f$ L \f$.
 * * A regra geral de otimização baseada em gradiente ajusta um parâmetro genérico \f$ \theta \f$
 * na direção oposta ao seu gradiente:
 * \f[ \theta_{t+1} = \theta_t - \eta \cdot \nabla_{\theta} L \f]
 * Onde \f$ \eta \f$ é a taxa de aprendizado. Algoritmos derivados (como Momentum, Adam)
 * estendem essa regra introduzindo conceitos físicos (inércia) ou estatísticos (momentos).
 */
class Optimizer {

protected:

    // @brief Taxa de aprendizado \f$ \eta \f$ (learning rate).
    /// * Hiperparâmetro crucial que controla o tamanho do passo dado na direção do gradiente
    /// negativo. Valores muito altos podem causar divergência, enquanto valores muito baixos
    /// tornam o treinamento excessivamente lento ou propício a prender em mínimos locais.
    float learning_rate_;

public:

    /**
     * @brief Construtor padrão do otimizador.
     * @param learning_rate Valor inicial para a taxa de aprendizado \f$ \eta \f$. O padrão é 0.01.
     */
    Optimizer(float learning_rate = 0.01f) : learning_rate_(learning_rate) {}

    /**
     * @brief Destrutor virtual padrão.
     * * Garante a liberação correta de memória caso otimizadores derivados
     * (que podem alocar matrizes de estado, como velocidades) sejam destruídos.
     */
    virtual ~Optimizer() = default;

    /**
     * @brief Atualiza os parâmetros de uma camada baseando-se nos gradientes calculados.
     * * Esta função aplica a regra de otimização específica para atualizar as matrizes
     * originais de pesos (\f$ W \f$) e vieses (\f$ b \f$). Em sua forma mais canônica
     * (Gradiente Descendente Padrão), a operação executada nas classes filhas será:
     * \f[ W = W - \eta \frac{\partial L}{\partial W} \f]
     * \f[ b = b - \eta \frac{\partial L}{\partial b} \f]
     * * @param weights Referência para a matriz de pesos da camada \f$ W \f$.
     * @param biases Referência para o vetor de vieses da camada \f$ b \f$.
     * @param grad_weights Referência constante para a matriz de gradientes dos pesos \f$ \frac{\partial L}{\partial W} \f$.
     * @param grad_biases Referência constante para o vetor de gradientes dos vieses \f$ \frac{\partial L}{\partial b} \f$.
     */
    virtual void Update (Eigen::MatrixXf &weights, Eigen::MatrixXf &biases, const Eigen::MatrixXf &grad_weights, const Eigen::MatrixXf &grad_biases) = 0;

};