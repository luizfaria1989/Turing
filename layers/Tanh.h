#pragma once
#include "../layers/Layer.h"

/**
 * @brief Implementa uma camada da função de ativação Tangente Hiperbólica (Tanh).
 * * Responsável por calcular a operação (e^(X) - e^(-X)) / (e^(X) + e^(-X))  onde X representa
 * a matriz de entrada.
 */
class Tanh : public Layer {

protected:
    /// @brief Matriz que armazena a saída do forward pass da tangente hiperbólica. É utilizada no cálculo do backward pass com intuito de evitar repetição de cálculos.
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
    * @brief Implementa o método forward pass da tanh.
    * @param input Referência constante para a matriz de entrada de dados.
    * @param output Referência para a matriz onde o resultado da camada será armazenado.
    * @note A fórmula matemática executada é: z = (e^(X) - e^(-X)) / (e^(X) + e^(-X)).
    * @warning As dimensões de 'input' devem ser (batch_size, input_size).
    */
    void Forward (const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        output = input.array().tanh();
        this->output_ = output;
    }

    /**
    * @brief Implementa o método backward pass da Tanh.
    * @param grad_input Referência para a matriz que armazenará o gradiente a ser enviado para a camada anterior.
    * @param grad_output Referência constante para o gradiente recebido da camada seguinte.
    * @note A fórmula matemática executada para o gradiente é: dX = dZ * (1 - tanh(X)^2).
    */
    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        grad_input = (grad_output.array() * (1.0 - this->output_.array().pow(2))).matrix();
    };

};