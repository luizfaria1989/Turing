#pragma once
#include "Optimizer.h"
#include <unordered_map>

/**
 * @class NAG
 * @brief Implementa o otimizador Nesterov Accelerated Gradient (NAG).
 * * Uma melhoria sobre o Gradiente Descendente com Momento. Enquanto o Momento padrão
 * calcula o gradiente na posição atual e então adiciona a inércia, o Nesterov
 * teoricamente calcula o gradiente em uma posição futura (lookahead), o que proporciona
 * uma correção de curso mais rápida e evita oscilações excessivas ao redor de mínimos.
 * * Como o cálculo do gradiente ocorre antes da etapa de otimização em nossa arquitetura,
 * implementamos a forma algebricamente rearranjada do NAG. Para um parâmetro genérico $\theta$,
 * velocidade $v$, taxa de aprendizado $\eta$ e momento $\gamma$, a atualização é:
 * \f[v_{t+1} = \gamma v_t + \eta \nabla_{\theta} L\f]
 * \f[\theta_{t+1} = \theta_t - (\gamma v_{t+1} + \eta \nabla_{\theta} L)\f]
 * * @note Essa formulação permite aplicar o conceito de "lookahead" do Nesterov
 * utilizando o gradiente calculado na posição atual \f$\theta_t\f$, mantendo o desacoplamento
 * entre o cálculo das derivadas (camadas) e a atualização (otimizador).
 */
class NAG : public Optimizer{

private:
    /// @brief Coeficiente de momento \f$\gamma\f$.
    /// Determina a influência da velocidade acumulada passada sobre a atualização atual.
    float momentum_;

    /// @brief Dicionário que armazena a velocidade (inércia) de cada matriz de parâmetros.
    /// O uso de ponteiros de memória constantes como chave garante que múltiplas camadas
    /// possam compartilhar o mesmo otimizador sem misturar seus estados de inércia.
    std::unordered_map<const Eigen::MatrixXf*, Eigen::MatrixXf> velocities_;

public:

    /**
     * @brief Construtor do otimizador Nesterov Accelerated Gradient.
     * @param learning_rate A taxa de aprendizado \f$\eta\f$. O padrão é 0.01.
     * @param momentum O coeficiente de momento \f$\gamma\f$. O padrão é 0.01.
     */
    NAG(float learning_rate = 0.01f, float momentum = 0.01f):
        Optimizer(learning_rate),
        momentum_ (momentum) {}

    /**
     * @brief Destrutor padrão.
     */
    virtual ~NAG() = default;

    /**
     * @brief Executa a atualização dos parâmetros utilizando a inércia de Nesterov.
     * * Semelhante ao Momento padrão, a velocidade é inicializada preguiçosamente (lazy initialization).
     * A atualização segue a variante vetorizada onde aplicamos a correção de Nesterov
     * diretamente utilizando o gradiente recém-calculado e a nova velocidade:
     * \f[v_{nova} = \gamma v_{atual} + \eta \frac{\partial L}{\partial W}\f]
     * \f[W = W - (\gamma v_{nova} + \eta \frac{\partial L}{\partial W})\f]
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

        // --- Atualização de Pesos ---
        // 1. Acumula a inércia e o gradiente na velocidade
        velocities_[&weights] = (momentum_ * velocities_[&weights]) + (learning_rate_ * grad_weights);
        // 2. Aplica o passo de Nesterov rearranjado
        weights = weights - (momentum_ * velocities_[&weights] + learning_rate_ * grad_weights);

        // --- Atualização de Vieses ---
        // 1. Acumula a inércia e o gradiente na velocidade
        velocities_[&biases] = (momentum_ * velocities_[&biases]) + (learning_rate_ * grad_biases);
        // 2. Aplica o passo de Nesterov rearranjado
        biases = biases - (momentum_ * velocities_[&biases] + learning_rate_ * grad_biases);

    };

};