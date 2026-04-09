#pragma once
#include <Eigen/Dense>

class Optimizer {
protected:
    float learning_rate_;

public:

    Optimizer(float learning_rate = 0.01f) : learning_rate_(learning_rate) {}

    virtual ~Optimizer() = default;

    virtual void Update (Eigen::MatrixXf &weights, Eigen::MatrixXf &biases, const Eigen::MatrixXf &grad_weights, const Eigen::MatrixXf &grad_biases) = 0;

};