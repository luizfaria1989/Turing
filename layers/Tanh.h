#pragma once
#include "../layers/Layer.h"

/**
 * @class Tanh
 * @brief Implementa a função de ativação Tangente Hiperbólica (Tanh).
 * * A função Tanh é uma versão reescalonada e deslocada da função Sigmoide.
 * Sua principal vantagem matemática é mapear os valores de entrada para o intervalo \f$ (-1, 1) \f$,
 * tornando as ativações centralizadas em zero (zero-centered). Isso reduz as oscilações
 * durante a atualização dos pesos e frequentemente resulta em uma convergência mais rápida.
 * A função matemática é definida como:
 * \f[ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \f]
 */
class Tanh : public Layer {

protected:

    /// @brief Cache da matriz de saída \f$ Y = \tanh(X) \f$ calculada no forward pass.
    /// @note Semelhante à Sigmoide, a derivada da Tanh pode ser expressa
    /// inteiramente em função da sua própria saída, permitindo um cálculo de gradiente otimizado
    /// de complexidade O(1) em relação a funções transcendentais no backward pass.
    Eigen::MatrixXf output_;

public:

    /**
     * @brief Construtor padrão da camada tanh.
     */
    Tanh() = default;

    /**
     * @brief Destrutor padrão.
     */
    virtual ~Tanh() = default;

    /**
    * @brief Executa o forward pass da ativação Tangente Hiperbólica.
    * * Aplica a função \f$ \tanh \f$ a cada elemento da matriz de entrada utilizando
    * a implementação otimizada da biblioteca Eigen.
    * \f[ Y = \tanh(X) \f]
    * * @param input Referência constante para a matriz de entrada \f$ X \f$ de dimensões \f$ (batch\_size \times input\_size) \f$.
    * @param output Referência para a matriz onde o resultado ativado \f$ Y \f$ será armazenado.
    */
    void Forward (const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        output = input.array().tanh();
        this->output_ = output;
    }

    /**
    * @brief Executa o backward pass da ativação Tangente Hiperbólica.
    * * A derivada matemática da função Tanh em relação à sua entrada \f$ X \f$ é dada por:
    * \f[ \tanh'(X) = 1 - \tanh^2(X) = 1 - Y^2 \f]
    * * Para propagar o erro, aplicamos a Regra da Cadeia multiplicando o gradiente
    * recebido \f$ \frac{\partial L}{\partial Y} \f$ pela derivada local através do
    * Produto de Hadamard (\f$ \odot \f$):
    * \f[ \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot (1 - Y^2) \f]
    * * @param grad_input Matriz que armazenará o gradiente calculado \f$ \frac{\partial L}{\partial X} \f$.
    * @param grad_output Matriz contendo o gradiente recebido da camada seguinte \f$ \frac{\partial L}{\partial Y} \f$.
    */
    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        grad_input = (grad_output.array() * (1.0 - this->output_.array().pow(2))).matrix();
    };

};