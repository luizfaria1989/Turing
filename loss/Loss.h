#pragma once
#include <Eigen/Dense>

/**
 * @class Loss
 * @brief Classe base abstrata para todas as funções de perda (Loss / Cost Functions).
 * * A função de perda atua como a métrica de erro objetivo da rede neural. Ela quantifica
 * a discrepância entre as previsões feitas pelo modelo e os rótulos reais (ground truth).
 * O objetivo de qualquer otimizador é minimizar este valor.
 * * De forma generalizada, para um lote (batch) de \f$N\f$ amostras, a perda agregada \f$L\f$
 * entre as previsões \f$\hat{Y}\f$ e os alvos reais \f$Y\f$ é dada por:
 * \f[L = \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, \hat{y}_i)\f]
 * Onde \f$\ell\f$ representa a função de erro individual (como Erro Quadrático Médio ou Entropia Cruzada).
 */
class Loss {

public:

    /**
     * @brief Construtor padrão da classe base de perda.
     */
    Loss() = default;

    /**
     * @brief Destrutor virtual padrão.
     * Garante a liberação correta de memória caso funções de perda filhas instanciem
     * estruturas de dados próprias no futuro.
     */
    virtual ~Loss() = default;

    /**
     * @brief Computa o valor escalar da perda (forward pass).
     * * Avalia o erro total do lote (batch) reduzindo as matrizes a um único valor escalar.
     * \f[L = f(\hat{Y}, Y)\f]
     * * @param predictions Matriz constante contendo as previsões geradas pelo modelo \f$\hat{Y}\f$. Suas dimensões geralmente são $(batch\_size \times num\_classes)$.
     * @param targets Matriz constante contendo os rótulos reais \f$Y\f$ correspondentes aos dados de entrada.
     * @param loss Referência para a variável escalar float onde o valor final calculado da perda \f$L\f$ será armazenado.
     */
    virtual void Forward (const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets, float &loss) = 0;

    /**
     * @brief Computa o gradiente da função de perda (backward pass).
     * * Este método é o ponto de partida exato do algoritmo de Backpropagation. Ele calcula a
     * derivada da função de perda em relação a cada elemento da previsão do modelo, gerando
     * a matriz Jacobiana que será propagada para a última camada da rede:
     * \f[\nabla_{\hat{Y}} L = \frac{\partial L}{\partial \hat{Y}}\f]
     * * @param predictions Matriz constante contendo as previsões do modelo \f$\hat{Y}\f$.
     * @param targets Matriz constante contendo os rótulos reais \f$Y\f$.
     * @param grad Matriz passada por referência que armazenará o gradiente calculado \f$\frac{\partial L}{\partial \hat{Y}}\f$. Terá exatamente as mesmas dimensões de \f$\hat{Y}\f$.
     */
    virtual void Backward (const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets, Eigen::MatrixXf &grad) = 0;

};