#pragma once
#include "Optimizer.h"
#include <unordered_map>

/**
 * @class GDMomentum
 * @brief Implementa o otimizador Gradiente Descendente com Momento (Momentum).
 * * Este algoritmo introduz o conceito físico de "inércia" na atualização dos parâmetros.
 * Em vez de depender apenas do gradiente atual, ele acumula uma média móvel exponencial
 * dos gradientes passados (velocidade). Isso ajuda a acelerar a convergência em direções
 * consistentes e a amortecer oscilações em ravinas muito íngremes da função de perda.
 * * A regra de atualização matemática para um parâmetro genérico \f$\theta\f$ e sua
 * respectiva velocidade \f$v\f$ é dada por:
 * \f[ v_{t+1} = \gamma v_t + \eta \nabla_{\theta} L\f]
 * \f[\theta_{t+1} = \theta_t - v_{t+1}\f]
 * Onde \f$\eta\f$ é a taxa de aprendizado e \f$\gamma\f$ é o coeficiente de momento.
 */
class GDMomentum : public Optimizer{

private:

    /// @brief Coeficiente de momento $\gamma$.
    /// Determina a contribuição dos gradientes passados para o passo atual.
    float momentum_;

    /// @brief Dicionário que armazena as velocidades (inércia) de cada matriz de parâmetros.
    /// Utiliza o endereço de memória constante da matriz de parâmetros como chave para
    /// garantir que cada camada rastreie sua própria inércia de forma independente.
    std::unordered_map<const Eigen::MatrixXf*, Eigen::MatrixXf> velocities_;

public:

    /**
     * @brief Construtor do otimizador GDMomentum.
     * @param learning_rate A taxa de aprendizado \f$\eta\f$. O padrão é 0.01.
     * @param momentum O coeficiente de momento \f$\gamma\f$.
     * @note Na literatura padrão de Deep Learning, valores comuns para o momento são maiores,
     * como 0.9 ou 0.99.
     */
    GDMomentum(float learning_rate = 0.01f, float momentum = 0.01f) :
        Optimizer(learning_rate),
        momentum_ (momentum) {}

    /**
     * @brief Destrutor padrão.
     */
    virtual ~GDMomentum() = default;

    /**
     * @brief Executa a atualização dos pesos e vieses utilizando inércia.
     * * Caso os parâmetros estejam sendo atualizados pela primeira vez (não existam no mapa
     * de velocidades), uma matriz de inércia zerada é inicializada preguiçosamente (lazy initialization).
     * Em seguida, as seguintes equações são aplicadas:
     * \f[v_W = \gamma v_W + \eta \frac{\partial L}{\partial W}\f]
     * \f[W = W - v_W\f]
     * (O mesmo é aplicado aos vieses \f$b\f$).
     * * @param weights Referência para a matriz de pesos \f$W\f$.
     * @param biases Referência para o vetor de vieses \f$b\f$.
     * @param grad_weights Matriz contendo os gradientes da perda em relação aos pesos \f$\frac{\partial L}{\partial W}\f$.
     * @param grad_biases Vetor contendo os gradientes da perda em relação aos vieses \f$\frac{\partial L}{\partial b}\f$.
     */
    void Update (Eigen::MatrixXf &weights, Eigen::MatrixXf &biases, const Eigen::MatrixXf &grad_weights, const Eigen::MatrixXf &grad_biases) override {

        // Inicialização preguiçosa da velocidade para os pesos (Lazy Initialization)
        if (velocities_.find(&weights) == velocities_.end()) {
            velocities_[&weights] = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        }

        // Inicialização preguiçosa da velocidade para os vieses
        if (velocities_.find(&biases) == velocities_.end()) {
            velocities_[&biases] = Eigen::MatrixXf::Zero(biases.rows(), biases.cols());
        }

        // Atualização da velocidade e aplicação nos pesos
        velocities_[&weights] = (momentum_ * velocities_[&weights]) + (learning_rate_ * grad_weights);
        weights = weights - velocities_[&weights];

        // Atualização da velocidade e aplicação nos vieses
        velocities_[&biases] = (momentum_ * velocities_[&biases]) + (learning_rate_ * grad_biases);
        biases = biases - velocities_[&biases];

    };

};