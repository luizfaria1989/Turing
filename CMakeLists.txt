cmake_minimum_required(VERSION 3.10) # Versão estável padrão
project(dnn_in_cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

add_executable(dnn_in_cpp
        main.cpp
        MNISTLoader.h
        layers/DenseLayer.h
        layers/Layer.h
        layers/ReLU.h
        layers/Sigmoid.h
        layers/Tanh.h
        layers/Softmax.h
        Loss/CategoricalCrossEntropy.h
        Loss/Loss.h
        Optimizers/Optimizer.h
        Optimizers/GradientDescent.h
        Optimizers/GDMomentum.h
        Optimizers/NAG.h
        Model/Model.h
        Metrics/Accuracy.h
)

find_library(ACCELERATE_FRAMEWORK Accelerate)

target_link_libraries(dnn_in_cpp PRIVATE Eigen3::Eigen ${ACCELERATE_FRAMEWORK})
target_compile_definitions(dnn_in_cpp PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACK)

add_executable(XORProblem models/XORProblem.cpp
        models/
        models/DNNWithNAG.cpp
        models/DNNWithGDMomentum.cpp)

add_executable(DNNWithGD models/DNNWithGD.cpp
        model/Model.h
        layers/DenseLayer.h
        layers/Sigmoid.h
        layers/Softmax.h
        loss/CategoricalCrossEntropy.h
        optimizers/GradientDescent.h
        MNISTLoader.h)

add_executable(DNNWithGDMomentum models/DNNWithGDMomentum.cpp
        model/Model.h
        layers/DenseLayer.h
        layers/Tanh.h
        layers/Softmax.h
        loss/CategoricalCrossEntropy.h
        optimizers/GDMomentum.h
        MNISTLoader.h)

add_executable(DNNWithNAG models/DNNWithNAG.cpp
        model/Model.h
        layers/DenseLayer.h
        layers/ReLU.h
        layers/Softmax.h
        loss/CategoricalCrossEntropy.h
        optimizers/NAG.h
        MNISTLoader.h)


target_link_libraries(XORProblem PRIVATE Eigen3::Eigen)
target_link_libraries(DNNWithGD PRIVATE Eigen3::Eigen)
target_link_libraries(DNNWithGDMomentum PRIVATE Eigen3::Eigen)
target_link_libraries(DNNWithNAG PRIVATE Eigen3::Eigen)