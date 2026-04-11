#pragma once
#include "../layers/Layer.h"

/**
 * @class ReLU
 * @brief Implementa a função de ativação Rectified Linear Unit (ReLU).
 * * A ReLU é uma função não-linear aplicada elemento a elemento na matriz de entrada.
 * É amplamente utilizada em redes neurais profundas por mitigar o problema do
 * desvanecimento do gradiente e induzir esparsidade nas ativações.
 * A função matemática é definida como:
 * \f[ f(x) = \max(0, x) \f]
 */
class ReLU : public Layer {

public:

    /**
    * @brief Construtor padrão da camada ReLU.
    * * Como esta camada não possui parâmetros treináveis (pesos ou vieses),
    * nenhuma inicialização de matriz é necessária.
    */
    ReLU() = default;

    /**
     * @brief Destrutor padrão.
     */
    virtual ~ReLU() = default;

    /**
    * @brief Executa o forward pass da ativação ReLU.
    * * Aplica a função limiar zero a todos os elementos da matriz de entrada.
    * \f[ Y = \max(0, X) \f]
    * * @param input Referência constante para a matriz de entrada \f$ X \f$ de dimensões \f$ (batch\_size \times input\_size) \f$.
    * @param output Referência para a matriz onde o resultado ativado \f$ Y \f$ será armazenado.
    */
    void Forward (const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        output = input.cwiseMax(0.0);
        this->input_ = input;
    }

    /**
    * @brief Executa o backward pass da ativação ReLU.
    * * A derivada da função ReLU em relação à sua entrada \f$ x \f$ é uma função indicadora:
    * \f[
    * f'(x) =
    * \begin{cases}
    * 1 & \text{se } x > 0 \\
    * 0 & \text{se } x \le 0
    * \end{cases}
    * \f]
    * Para propagar o erro, aplicamos a Regra da Cadeia multiplicando o gradiente
    * recebido \f$ \frac{\partial L}{\partial Y} \f$ pela derivada local utilizando o
    * Produto de Hadamard (\f$ \odot \f$):
    * \f[ \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot f'(X) \f]
    * * @param grad_input Referência para a matriz que armazenará o gradiente calculado \f$ \frac{\partial L}{\partial X} \f$.
    * @param grad_output Matriz contendo o gradiente recebido da camada seguinte \f$ \frac{\partial L}{\partial Y} \f$.
    */
    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        // .array() permite operações elemento a elemento (Produto de Hadamard)
        grad_input = (grad_output.array() * (this->input_.array() > 0.0).cast<float>()).matrix();
    };

};