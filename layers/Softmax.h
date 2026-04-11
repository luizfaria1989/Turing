#pragma once
#include "../layers/Layer.h"

/**
 * @class Softmax
 * @brief Implementa a função de ativação Softmax.
 * * A função Softmax é tipicamente utilizada na última camada de redes neurais
 * de classificação multiclasse. Ela converte um vetor de números reais (logits)
 * em uma distribuição de probabilidade, garantindo que todas as saídas estejam
 * no intervalo \f$ (0, 1) \f$ e que a soma das probabilidades de todas as classes seja igual a 1.
 * A função matemática para o i-ésimo elemento é:
 * \f[ y_i = \frac{e^{x_i}}{\sum_{j=1}^{C} e^{x_j}} \f]
 * Onde \f$ C \f$ é o número total de classes (neurônios na camada).
 */
class Softmax : public Layer {

protected:
    /// @brief Cache da matriz de probabilidades \f$ Y \f$ calculada no forward pass.
    /// @note Assim como na Sigmoide, o cálculo do gradiente da Softmax pode ser
    /// computado inteiramente a partir de sua saída, evitando o recálculo de exponenciais.
    Eigen::MatrixXf output_;


public:

    /**
     * @brief Construtor padrão da camada Softmax.
     */
    Softmax() = default;

    /**
     * @brief Destrutor padrão.
     */
    virtual ~Softmax() = default;

    /**
     * @brief Executa o forward pass da ativação Softmax.
     * * Aplica a exponenciação a cada elemento e normaliza dividindo pela soma
     * das exponenciais da respectiva amostra (linha).
     * * @param input Referência constante para a matriz de logits \f$ X \f$ de dimensões \f$ (batch\_size \times num\_classes) \f$.
     * @param output Referência para a matriz onde a distribuição de probabilidade \f$ Y \f$ será armazenada.
     */
    void Forward (const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        Eigen::MatrixXf exp_values = input.array().exp();
        output = exp_values.array().colwise() / (exp_values.rowwise().sum()).array();
        this->output_ = output;
    }

    /**
     * @brief Executa o backward pass da ativação Softmax.
     * * Como a saída \f$ y_i \f$ depende de todos os \f$ x_j \f$, a derivada real da
     * Softmax é uma matriz Jacobiana. Para evitar o custo de memória de instanciar
     * uma matriz tridimensional para o batch, aplicamos o "Jacobian-vector product" diretamente.
     * * A equação vetorizada do gradiente para uma amostra é dada por:
     * \f[ \frac{\partial L}{\partial X} = Y \odot \left( \frac{\partial L}{\partial Y} - \left( \sum_{j} \frac{\partial L}{\partial Y_j} \cdot Y_j \right) \mathbf{1} \right) \f]
     * * @param grad_input Matriz que armazenará o gradiente calculado \f$ \frac{\partial L}{\partial X} \f$.
     * @param grad_output Matriz contendo o gradiente recebido da camada seguinte (geralmente a função de perda) \f$ \frac{\partial L}{\partial Y} \f$.
     */
    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        grad_input = (this->output_.array() * (grad_output.array().colwise() - (grad_output.array() * this->output_.array()).rowwise().sum())).matrix();
    };

};