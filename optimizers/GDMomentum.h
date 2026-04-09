#pragma once
#include "Optimizer.h"
#include <unordered_map>

class GDMomentum : public Optimizer{

private:
    float momentum_;
    std::unordered_map<const Eigen::MatrixXf*, Eigen::MatrixXf> velocities_;

public:

    GDMomentum(float learning_rate = 0.01f, float momentum = 0.01f) : Optimizer(learning_rate), momentum_ (momentum) {}

    virtual ~GDMomentum() = default;

    void Update (Eigen::MatrixXf &weights, Eigen::MatrixXf &biases, const Eigen::MatrixXf &grad_weights, const Eigen::MatrixXf &grad_biases) override {

        if (velocities_.find(&weights) == velocities_.end()) {
            velocities_[&weights] = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());
        }

        if (velocities_.find(&biases) == velocities_.end()) {
            velocities_[&biases] = Eigen::MatrixXf::Zero(biases.rows(), biases.cols());
        }

        velocities_[&weights] = (momentum_ * velocities_[&weights]) + (learning_rate_ * grad_weights);
        weights = weights - velocities_[&weights];

        velocities_[&biases] = (momentum_ * velocities_[&biases]) + (learning_rate_ * grad_biases);
        biases = biases - velocities_[&biases];

    };

};