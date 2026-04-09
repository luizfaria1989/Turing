#pragma once
#include <Eigen/Dense>
#include "../optimizers/Optimizer.h"

class Layer {
protected:
    Eigen::MatrixXf input_;

public:

    virtual void Forward(const Eigen::MatrixXf &input, Eigen::MatrixXf &output) = 0;

    virtual void Backward(Eigen::MatrixXf &grad_input, const Eigen::MatrixXf &grad_output) = 0;

    virtual void UpdateParams(Optimizer* optimizer) {}

    virtual void SaveParams(std::ofstream &file){}

};