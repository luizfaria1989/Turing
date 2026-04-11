#pragma once
#include <Eigen/Dense>
#include "../optimizers/Optimizer.h"

/**
 * @class Layer
 * @brief Classe base abstrata para todas as camadas da rede neural.
 * * Define o contrato polimórfico que qualquer componente da rede (camadas densas,
 * convolucionais, funções de ativação) deve implementar. Ela garante que o
 * modelo possa orquestrar o fluxo de dados em lote (batch) de forma genérica.
 */
class Layer {
protected:

    /// @brief Cache da matriz de entrada recebida durante o forward pass.
    /// * Armazenar o estado \f$ X \f$ é essencial para o cálculo das derivadas
    /// analíticas durante o backward pass.
    Eigen::MatrixXf input_;

public:

    /**
     * @brief Destrutor virtual padrão.
     * * Garante a liberação correta de memória das classes derivadas quando
     * destruídas a partir de um ponteiro da classe base.
     */
    virtual ~Layer(){}

    /**
     * @brief Executa o forward pass (propagação direta) da camada.
     * * Aplica a transformação matemática específica da camada aos dados de entrada.
     * De forma generalizada, computa:
     * \f[ Y = f(X) \f]
     * Onde \f$ X \f$ é a entrada e \f$ f \f$ é a operação interna da camada.
     * * @param input Referência constante para a matriz de entrada \f$ X \f$.
     * @param output Referência para a matriz onde o resultado \f$ Y \f$ será armazenado.
     */
    virtual void Forward(const Eigen::MatrixXf &input, Eigen::MatrixXf &output) = 0;

    /**
     * @brief Executa o backward pass (retropropagação) da camada.
     * * Aplica a Regra da Cadeia para propagar o gradiente do erro.
     * A camada recebe o gradiente da perda em relação à sua saída, \f$ \frac{\partial L}{\partial Y} \f$,
     * e calcula o gradiente em relação à sua entrada, \f$ \frac{\partial L}{\partial X} \f$,
     * propagando-o para a camada anterior:
     * \f[ \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial X} \f]
     * * @param grad_input Matriz que armazenará o gradiente calculado \f$ \frac{\partial L}{\partial X} \f$.
     * @param grad_output Matriz contendo o gradiente recebido da camada seguinte \f$ \frac{\partial L}{\partial Y} \f$.
     */
    virtual void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) = 0;

    /**
     * @brief Atualiza os parâmetros internos da camada (pesos, vieses).
     * * Camadas parametrizadas (ex: DenseLayer) devem sobrescrever este método
     * para aplicar os gradientes calculados utilizando o otimizador fornecido.
     * * @note Camadas sem parâmetros treináveis (ex: ReLU, Softmax) utilizam
     * esta implementação base vazia, ignorando o passo de atualização.
     * * @param optimizer Ponteiro para o otimizador que dita a regra de atualização.
     */
    virtual void UpdateParams(Optimizer* optimizer) {}

    /**
     * @brief Serializa os parâmetros da camada em disco.
     * * Implementação base vazia. Camadas com estado interno de aprendizado
     * devem sobrescrever este método para persistir suas matrizes.
     * * @param file Referência para o fluxo de arquivo de saída onde os dados serão gravados.
     */
    virtual void SaveParams(std::ofstream &file){}

};