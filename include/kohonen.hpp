#pragma once

#include "neuron.hpp"
#include <vector>

class KohonenSOM
{
    int width, height, inputDim;
    float learningRate, radius;
    int epochs;
    std::vector<Neuron> neurons;

    float euclideanDistance(const std::vector<float> &a, const std::vector<float> &b) const;
    float neighborhoodFunction(int dx, int dy, float radius) const;
    void updateWeights(Neuron &neuron, const std::vector<float> &input, float lr, float influence);
    Neuron *findBMU(const std::vector<float> &input);

public:
    KohonenSOM(int inputDim, int width, int height, int epochs = 10, float learningRate = 0.1f);

    void train(const std::vector<std::vector<float>> &data,
               const std::vector<std::vector<float>> &test_data);
    void printBMUPosition(const std::vector<float> &input) const;
    void printMapActivity() const;
};
