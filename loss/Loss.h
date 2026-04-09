#pragma once
#include <Eigen/Dense>

class Loss {

public:

    Loss() = default;

    virtual ~Loss() = default;

    virtual void Forward (const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets, float &loss) = 0;

    virtual void Backward (const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets, Eigen::MatrixXf &grad) = 0;

};