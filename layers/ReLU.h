#pragma once
#include "../layers/Layer.h"

/**
 * @brief Implementa uma camada da função de ativação Rectified Linear Unit (ReLU).
 * * Responsável por calcular a operação max(0, X) onde X representa
 * a matriz de entrada.
 */
class ReLU : public Layer {

public:

    /**
     * @brief Construtor padrão da camada ReLU.
     */
    ReLU() = default;

    /**
     * @brief Destrutor padrão.
     */
    virtual ~ReLU() = default;

    /**
    * @brief Implementa o método forward pass da ReLU.
    * @param input Referência constante para a matriz de entrada de dados.
    * @param output Referência para a matriz onde o resultado da camada será armazenado.
    * @note A fórmula matemática executada é: Z = max(0, X).
    * @warning As dimensões de 'input' devem ser (batch_size, input_size).
    */
    void Forward (const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        output = input.cwiseMax(0.0);
        this->input_ = input;
    }

    /**
    * @brief Implementa o método backward pass da ReLU.
    * @param grad_input Referência para a matriz que armazenará o gradiente a ser enviado para a camada anterior.
    * @param grad_output Referência constante para o gradiente recebido da camada seguinte.
    * @note A fórmula matemática executada para o gradiente é: dX = dZ * (X > 0 ? 1 : 0).
    */
    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        grad_input = (grad_output.array() * (this->input_.array() > 0.0).cast<float>()).matrix();
    };

};