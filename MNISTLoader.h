#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

class MNISTLoader {
private:
    // Truque mágico para inverter os bytes (Big-Endian para Little-Endian)
    static uint32_t swapEndian(uint32_t val) {
        return ((val << 24) & 0xff000000) |
               ((val <<  8) & 0x00ff0000) |
               ((val >>  8) & 0x0000ff00) |
               ((val >> 24) & 0x000000ff);
    }

public:
    // Carrega as imagens e já normaliza dividindo por 255.0
    static Eigen::MatrixXf loadImages(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Erro ao abrir arquivo de imagens!");

        uint32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_images, sizeof(num_images));
        file.read((char*)&num_rows, sizeof(num_rows));
        file.read((char*)&num_cols, sizeof(num_cols));

        num_images = swapEndian(num_images);
        num_rows = swapEndian(num_rows);
        num_cols = swapEndian(num_cols);

        int image_size = num_rows * num_cols; // 28x28 = 784
        Eigen::MatrixXf images(num_images, image_size);

        for (int i = 0; i < num_images; ++i) {
            for (int j = 0; j < image_size; ++j) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, sizeof(pixel));
                images(i, j) = static_cast<float>(pixel) / 255.0f; // Normalização!
            }
        }
        return images;
    }

    // Carrega os gabaritos e aplica o One-Hot Encoding (ex: 3 vira [0 0 0 1 0 0 0 0 0 0])
    static Eigen::MatrixXf loadLabels(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Erro ao abrir arquivo de labels!");

        uint32_t magic_number = 0, num_items = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_items, sizeof(num_items));

        num_items = swapEndian(num_items);

        Eigen::MatrixXf labels = Eigen::MatrixXf::Zero(num_items, 10); // 10 classes (0 a 9)

        for (int i = 0; i < num_items; ++i) {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels(i, static_cast<int>(label)) = 1.0f; // Marca 1.0 na coluna correta
        }
        return labels;
    }
};