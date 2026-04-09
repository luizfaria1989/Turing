#pragma once
#include "../loss/Loss.h"

class CategoricalCrossEntropy : public Loss {

public:

    CategoricalCrossEntropy() = default;

    virtual ~CategoricalCrossEntropy() = default;

    void Forward (const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets, float &loss)  {
        loss = -((targets.array() * (predictions.array() + 1e-7f).log()).sum())/predictions.rows();
    }

    void Backward (const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets, Eigen::MatrixXf &grad) {
        grad = (-1.0f/predictions.rows() * (targets.array() / (predictions.array() + 1e-7f))).matrix();
    }

};