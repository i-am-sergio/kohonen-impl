cmake_minimum_required(VERSION 3.10)
project(perceptron)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# Ejecutable de entrenamiento
add_executable(kohonen
     main.cpp
     src/kohonen.cpp
     src/neuron.cpp
)