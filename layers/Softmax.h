#pragma once
#include "../layers/Layer.h"

class Softmax : public Layer {

protected:
    Eigen::MatrixXf output_;


public:

    Softmax() = default;

    virtual ~Softmax() = default;

    void Forward (const Eigen::MatrixXf &input, Eigen::MatrixXf &output) override {
        Eigen::MatrixXf exp_values = input.array().exp();
        output = exp_values.array().colwise() / (exp_values.rowwise().sum()).array();
        this->output_ = output;
    }

    void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) override {
        grad_input = (this->output_.array() * (grad_output.array().colwise() - (grad_output.array() * this->output_.array()).rowwise().sum())).matrix();
    };

};