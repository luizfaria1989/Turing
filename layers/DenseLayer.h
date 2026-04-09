#pragma once
#include "Layer.h"

/**
 * @brief Implementa uma camada densa da rede neural.
 * * Responsável por calcular a transformação linear Z = X * W + b, onde X representa
 * a matriz de entrada, W a matriz de pesos e b o vetor de vieses da camada.
 */
class DenseLayer : public Layer {

protected:
    /// @brief Matriz de pesos da camada, atualizada durante o backward pass.
    Eigen::MatrixXf weights_;
    /// @brief Vetor de vieses da camada, atualizado durante o backward pass.
    Eigen::MatrixXf biases_;
    /// @brief Matriz contendo os gradientes dos pesos, utilizada para atualizar a matriz de pesos original.
    Eigen::MatrixXf grad_weights_;
    /// @brief Matriz contentendo os gradientes dos vieses, é utilizada para atualizar o vetor de vieses.
    Eigen::MatrixXf grad_biases_;
    /// @brief Quantidade de neurônios na camada.
    int neurons_;

public:

    /**
     * @brief Construtor padrão da camada densa. Inicializa os pesos aleatoriamente e os vieses com zero.
     * @param input_size Número de características (features) da entrada vinda da camada anterior.
     * @param neurons Número de neurônios que esta camada terá.
     */
    DenseLayer(const int input_size, const int neurons): weights_(Eigen::MatrixXf::Random(input_size, neurons) * 0.01f), biases_(Eigen::MatrixXf::Zero(1, neurons)), grad_weights_(Eigen::MatrixXf::Zero(input_size, neurons)), grad_biases_(Eigen::MatrixXf::Zero(1, neurons)), neurons_(neurons){}

    /**
     * @brief Destrutor padrão.
     */
    virtual ~DenseLayer() = default;

    /**
    * @brief Implementa o método forward pass da camada densa.
    * @param input Referência constante para a matriz de entrada de dados.
    * @param output Referência para a matriz onde o resultado da camada será armazenado.
    * @note A fórmula matemática executada é: Z = X * W + b.
    * @warning As dimensões de 'input' devem ser (batch_size, input_size).
    */
    void Forward(const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        output = (input * weights_).array().rowwise() + biases_.row(0).array();
        this->input_ = input;
    }

    /**
    * @brief Implementa o método backward pass da camada densa.
    * * Calcula a matriz de gradientes dos pesos, dos vieses e o gradiente que será propagado.
    * @param grad_input Referência para a matriz que armazenará o gradiente a ser enviado para a camada anterior.
    * @param grad_output Referência constante para o gradiente recebido da camada seguinte.
    */
    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        grad_weights_ = this->input_.transpose() * grad_output;
        grad_biases_ = grad_output.colwise().sum();
        grad_input = grad_output * this->weights_.transpose();
    };

    /**
    * @brief Atualiza os parâmetros (pesos e vieses) da camada utilizando o otimizador.
    * @param optimizer Ponteiro para o otimizador escolhido na construção do modelo.
    * @see Optimizer::Update()
    */
    void UpdateParams(Optimizer* optimizer) override {
        optimizer->Update(this->weights_, this->biases_, this->grad_weights_, this->grad_biases_);
    }

    /**
    * @brief Salva os parâmetros atuais da camada em um arquivo de texto.
    * @param file Referência para o fluxo de arquivo (ofstream) onde os dados serão gravados.
    */
    void SaveParams(std::ofstream &file) override {
        file << this->weights_ << "\n" << this->biases_ << "\n";
    }

};