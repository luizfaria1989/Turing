#pragma once
#include "../layers/Layer.h"

/**
 * @class Sigmoid
 * @brief Implementa a função de ativação Sigmoide Logística.
 * * A função sigmoide mapeia qualquer valor real para o intervalo \f$ (0, 1) \f$.
 * Historicamente, é muito utilizada para modelar probabilidades e transformar
 * saídas lineares em distribuições. A função matemática é definida como:
 * \f[ \sigma(x) = \frac{1}{1 + e^{-x}} \f]
 */
class Sigmoid : public Layer {

protected:
    /// @brief Cache da matriz de saída \f$ Y = \sigma(X) \f$ calculada no forward pass.
    /// @note Armazenar a saída (em vez da entrada) é uma otimização algorítmica fundamental.
    /// Como a derivada da sigmoide pode ser expressa em função da sua própria saída,
    /// evitamos recalcular a operação exponencial durante a retropropagação.
    Eigen::MatrixXf output_;

public:

    /**
    * @brief Construtor padrão da camada Sigmoid.
    * Como esta camada não possui parâmetros treináveis (pesos/vieses),
    * a inicialização ocorre sem alocação de matrizes.
    */
    Sigmoid() = default;

    /**
     * @brief Destrutor padrão.
     */
    virtual ~Sigmoid() = default;

    /**
    * @brief Executa o forward pass da ativação Sigmoide.
    * * Aplica a função logística a cada elemento da matriz de entrada.
    * \f[ Y = \frac{1}{1 + e^{-X}} \f]
    * * @param input Referência constante para a matriz de entrada \f$ X \f$ de dimensões \f$ (batch\_size \times input\_size) \f$.
    * @param output Referência para a matriz onde o resultado ativado \f$ Y \f$ será armazenado.
    */
    void Forward (const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        output = 1.0 / (1.0 + (-input.array()).exp());
        this->output_ = output;
    }

    /**
    * @brief Executa o backward pass da ativação Sigmoide.
    * * Uma das propriedades matemáticas mais elegantes da função sigmoide é que sua
    * derivada pode ser calculada diretamente a partir de sua saída \f$ Y \f$:
    * \f[ \sigma'(X) = Y \odot (1 - Y) \f]
    * * Para propagar o erro, aplicamos a Regra da Cadeia multiplicando o gradiente
    * recebido \f$ \frac{\partial L}{\partial Y} \f$ pela derivada local através do
    * Produto de Hadamard (\f$ \odot \f$):
    * \f[ \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot (Y \odot (1 - Y)) \f]
    * * @param grad_input Matriz que armazenará o gradiente calculado \f$ \frac{\partial L}{\partial X} \f$.
    * @param grad_output Matriz contendo o gradiente recebido da camada seguinte \f$ \frac{\partial L}{\partial Y} \f$.
    */
    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        // Utiliza o cache output_ para um cálculo de gradiente O(1) em relação a funções transcendentais.
        grad_input = (grad_output.array() * (this->output_.array() * (1.0 - this->output_.array()))).matrix();
    };

};