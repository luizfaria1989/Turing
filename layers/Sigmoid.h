#pragma once
#include "../layers/Layer.h"

/**
 * @brief Implementa uma camada da função de ativação Sigmoide Logística (Logistic Sigmoid).
 * * Responsável por calcular a operação 1 (1 + e^(-X)) onde X representa
 * a matriz de entrada.
 */
class Sigmoid : public Layer {

protected:
    /// @brief Matriz que armazena a saída do forward pass da sigmoide logística. É utilizada no cálculo do backward pass com intuito de evitar repetição de cálculos.
    Eigen::MatrixXf output_;

public:

    /**
     * @brief Construtor padrão da camada Sigmoid.
     */
    Sigmoid() = default;

    /**
     * @brief Destrutor padrão.
     */
    virtual ~Sigmoid() = default;

    /**
    * @brief Implementa o método forward pass da Sigmoid.
    * @param input Referência constante para a matriz de entrada de dados.
    * @param output Referência para a matriz onde o resultado da camada será armazenado.
    * @note A fórmula matemática executada é: z= 1 / (1 + e^(-X)).
    * @warning As dimensões de 'input' devem ser (batch_size, input_size).
    */
    void Forward (const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        output = 1.0 / (1.0 + (-input.array()).exp());
        this->output_ = output;
    }

    /**
    * @brief Implementa o método backward pass da Sigmoid.
    * @param grad_input Referência para a matriz que armazenará o gradiente a ser enviado para a camada anterior.
    * @param grad_output Referência constante para o gradiente recebido da camada seguinte.
    * @note A fórmula matemática executada para o gradiente é: dX = dZ * sigma(X) * (1 - sigma(X)).
    */
    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        grad_input = (grad_output.array() * (this->output_.array() * (1.0 - this->output_.array()))).matrix();
    };

};