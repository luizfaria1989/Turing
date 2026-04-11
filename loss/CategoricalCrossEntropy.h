#pragma once
#include "../loss/Loss.h"

/**
 * @class CategoricalCrossEntropy
 * @brief Implementa a função de perda Entropia Cruzada Categórica (Categorical Cross-Entropy).
 * * Amplamente utilizada em problemas de classificação multiclasse (junto à ativação Softmax).
 * Ela mede a divergência entre a distribuição de probabilidade prevista pelo modelo e a
 * distribuição real dos dados (geralmente representada em one-hot encoding).
 * * A perda média para um lote (batch) de $N$ amostras e \f$C\f$ classes é dada por:
 * \f[L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} Y_{i,c} \log(\hat{Y}_{i,c} + \epsilon)\f]
 * Onde:
 * - \f$Y\f$ é a matriz de alvos reais (targets).
 * - \f$\hat{Y}\f$ é a matriz de previsões (predictions).
 * - \f$\epsilon\f$ é um valor infinitesimal somado para garantir estabilidade numérica.
 */
class CategoricalCrossEntropy : public Loss {

public:

    /**
     * @brief Construtor padrão da função de perda Entropia Cruzada Categórica.
     */
    CategoricalCrossEntropy() = default;

    /**
     * @brief Destrutor padrão.
     */
    virtual ~CategoricalCrossEntropy() = default;

    /**
     * @brief Computa o valor escalar da perda (forward pass).
     * * Aplica o logaritmo natural elemento a elemento nas previsões, pondera pelos
     * alvos reais, e calcula a média ao longo de todo o lote (batch size).
     * * @note Adicionamos \f$\epsilon = 10^{-7}\f$ (\f$\text{1e-7f}\f$) dentro do logaritmo para evitar a
     * indefinição matemática e falha computacional ($\text{NaN}$) ao calcular \f$\log(0)\f$.
     * * @param predictions Matriz constante contendo as probabilidades previstas \f$\hat{Y}\f$.
     * @param targets Matriz constante contendo os rótulos reais $Y$.
     * @param loss Referência para a variável escalar onde o erro médio $L$ será armazenado.
     */
    void Forward (const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets, float &loss)  override {
        loss = -((targets.array() * (predictions.array() + 1e-7f).log()).sum())/predictions.rows();
    }

    /**
     * @brief Computa o gradiente da Entropia Cruzada Categórica (backward pass).
     * * Calcula a derivada da função de perda em relação a cada previsão feita pelo modelo.
     * A matriz resultante representa o ponto de partida do algoritmo de backpropagation.
     * A derivada matemática vetorizada (incluindo o fator de média do batch) é:
     * \f[\frac{\partial L}{\partial \hat{Y}} = -\frac{1}{N} \left( \frac{Y}{\hat{Y} + \epsilon} \right)\f]
     * * @param predictions Matriz constante contendo as previsões do modelo \f$\hat{Y}\f$.
     * @param targets Matriz constante contendo os rótulos reais \f$Y\f$.
     * @param grad Matriz passada por referência que armazenará o gradiente calculado \f$\frac{\partial L}{\partial \hat{Y}}\f$.
     */
    void Backward (const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets, Eigen::MatrixXf &grad) override {
        grad = (-1.0f/predictions.rows() * (targets.array() / (predictions.array() + 1e-7f))).matrix();
    }

};