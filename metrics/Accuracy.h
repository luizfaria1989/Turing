#pragma once
#include <Eigen/Dense>


    float CalculateAccuracy(const Eigen::MatrixXf &predictions, const Eigen::MatrixXf &targets) {
        int correct_predictions = 0;
        int num_samples = predictions.rows();

        for (int i = 0; i < num_samples; ++i) {
            Eigen::Index pred_index;
            Eigen::Index target_index;

            predictions.row(i).maxCoeff(&pred_index);

            targets.row(i).maxCoeff(&target_index);

            if (pred_index == target_index) {
                correct_predictions++;
            }
        }

        return static_cast<float>(correct_predictions) / num_samples;
    };