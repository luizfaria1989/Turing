#pragma once
#include "Layer.h"

/**
 * @class DenseLayer
 * @brief Implementa uma camada densa (fully connected) da rede neural.
 * * Responsável por aplicar uma transformação linear aos dados de entrada.
 * A operação fundamental desta camada no forward pass é:
 * \f[ Z = X \cdot W + b \f]
 * Onde:
 * - \f$ X \f$ é a matriz de entrada.
 * - \f$ W \f$ é a matriz de pesos.
 * - \f$ b \f$ é o vetor de vieses (transmitido para todas as linhas via broadcasting).
 */
class DenseLayer : public Layer {

protected:

    /// @brief Matriz de pesos da camada \f$ (input\_size \times neurons) \f$.
    Eigen::MatrixXf weights_;

    /// @brief Vetor de vieses da camada \f$ (1 \times neurons) \f$.
    Eigen::MatrixXf biases_;

    /// @brief Gradientes da função de perda em relação aos pesos \f$ \frac{\partial L}{\partial W} \f$.
    Eigen::MatrixXf grad_weights_;

    /// @brief Gradientes da função de perda em relação aos vieses \f$ \frac{\partial L}{\partial b} \f$.
    Eigen::MatrixXf grad_biases_;

    /// @brief Quantidade de neurônios na camada (dimensão de saída).
    int neurons_;

public:

    /**
    * @brief Construtor da camada densa.
    * * Inicializa os pesos a partir de uma distribuição uniforme (escalonada por 0.01 para evitar saturação inicial)
    * e os vieses com zeros.
    * * @param input_size Número de características (features) da entrada.
    * @param neurons Número de neurônios da camada (tamanho da saída).
    */
    DenseLayer(const int input_size, const int neurons):
        weights_(Eigen::MatrixXf::Random(input_size, neurons) * 0.7f),
        biases_(Eigen::MatrixXf::Zero(1, neurons)),
        grad_weights_(Eigen::MatrixXf::Zero(input_size, neurons)),
        grad_biases_(Eigen::MatrixXf::Zero(1, neurons)),
        neurons_(neurons){}


    /**
     * @brief Destrutor padrão.
     */
    virtual ~DenseLayer() = default;

    /**
     * @brief Executa o forward pass da camada densa.
     * * @param input Referência constante para a matriz de entrada de dimensões \f$ (batch\_size \times input\_size) \f$.
     * @param output Referência para a matriz onde o resultado \f$ Z \f$ será armazenado, de dimensões \f$ (batch\_size \times neurons) \f$.
     * * @note Utiliza broadcasting do Eigen (`rowwise()`) para somar o vetor de vieses a cada amostra do batch.
     */
    void Forward(const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        output = (input * weights_).array().rowwise() + biases_.row(0).array();
        this->input_ = input;
    }

    /**
    * @brief Executa o backward pass, calculando os gradientes locais e propagando o erro.
    * * A partir do gradiente recebido da camada posterior \f$ \frac{\partial L}{\partial Z} \f$,
    * esta função calcula as seguintes derivadas parciais:
    * * 1. Gradiente dos pesos: \f[ \frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Z} \f]
    * 2. Gradiente dos vieses: \f[ \frac{\partial L}{\partial b} = \sum_{i=1}^{m} \left( \frac{\partial L}{\partial Z} \right)_i \f]
    * 3. Gradiente da entrada (propagado para trás): \f[ \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot W^T \f]
    * * @param grad_input Matriz que armazenará o gradiente a ser enviado para a camada anterior \f$ (batch\_size \times input\_size) \f$.
    * @param grad_output Gradiente da perda em relação à saída desta camada \f$ (batch\_size \times neurons) \f$.
    */
    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        grad_weights_ = this->input_.transpose() * grad_output;
        grad_biases_ = grad_output.colwise().sum();
        grad_input = grad_output * this->weights_.transpose();
    };

    /**
    * @brief Atualiza os pesos e vieses da camada.
    * * @param optimizer Ponteiro para o otimizador que aplicará a regra de atualização (ex: SGD, Momentum, Adam).
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