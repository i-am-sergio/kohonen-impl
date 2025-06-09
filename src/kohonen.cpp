#include "kohonen.hpp"
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <ctime>

KohonenSOM::KohonenSOM(int inputDim, int width, int height, int epochs, float learningRate)
    : inputDim(inputDim), width(width), height(height), epochs(epochs),
      learningRate(learningRate), radius(std::max(width, height) / 2.0f)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    // Xavier/Glorot initialization
    double limit = std::sqrt(6.0 / (inputDim));
    std::uniform_real_distribution<float> dis(-limit, limit);

    neurons.reserve(width * height);
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
            neurons.emplace_back(inputDim, i, j, gen, dis);
}

float KohonenSOM::euclideanDistance(const std::vector<float> &a, const std::vector<float> &b) const
{
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(dist);
}

float KohonenSOM::neighborhoodFunction(int dx, int dy, float radius) const
{
    float dist2 = static_cast<float>(dx * dx + dy * dy);
    return std::exp(-dist2 / (2 * radius * radius));
}

void KohonenSOM::updateWeights(Neuron &neuron, const std::vector<float> &input, float lr, float influence)
{
    for (size_t i = 0; i < neuron.weights.size(); ++i)
        neuron.weights[i] += lr * influence * (input[i] - neuron.weights[i]);
}

Neuron *KohonenSOM::findBMU(const std::vector<float> &input)
{
    int bmu_idx = 0;
    float minDist = euclideanDistance(input, neurons[0].weights);

#pragma omp parallel
    {
        float localMinDist = minDist;
        int localBmuIdx = bmu_idx;

#pragma omp for nowait
        for (int i = 1; i < neurons.size(); ++i)
        {
            float dist = euclideanDistance(input, neurons[i].weights);
            if (dist < localMinDist)
            {
                localMinDist = dist;
                localBmuIdx = i;
            }
        }

#pragma omp critical
        {
            if (localMinDist < minDist)
            {
                minDist = localMinDist;
                bmu_idx = localBmuIdx;
            }
        }
    }

#pragma omp atomic
    neurons[bmu_idx].bmu_count++;
    return &neurons[bmu_idx];
}

void KohonenSOM::train(const std::vector<std::vector<float>> &data,
                       const std::vector<std::vector<float>> &test_data)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::clock_t start = std::clock();
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;

        for (const auto &input : data)
        {
            Neuron *bmu = findBMU(input);

#pragma omp parallel for
            for (int i = 0; i < neurons.size(); ++i)
            {
                int dx = neurons[i].x - bmu->x;
                int dy = neurons[i].y - bmu->y;
                float influence = neighborhoodFunction(dx, dy, radius);
                updateWeights(neurons[i], input, learningRate, influence);
            }
        }

        learningRate *= 0.99f;
        radius *= 0.99f;
        // Medir tiempo
        std::clock_t end = std::clock();
        double duration = double(end - start) / CLOCKS_PER_SEC;
        std::cout << "Epoch duration: " << duration << " seconds\n";

        // Validación (muy básica: distancia promedio al BMU)
        double total_distance = 0.0;
        for (const auto &input : test_data)
        {
            Neuron *bmu = findBMU(input);
            total_distance += euclideanDistance(input, bmu->weights);
        }
        double avg_distance = total_distance / test_data.size();
        std::cout << "Validation average BMU distance: " << avg_distance << "\n\n";
    }
}

void KohonenSOM::printBMUPosition(const std::vector<float> &input) const
{
    const Neuron *bmu = nullptr;
    float minDist = 1e9f;
    for (const auto &neuron : neurons)
    {
        float dist = euclideanDistance(input, neuron.weights);
        if (dist < minDist)
        {
            minDist = dist;
            bmu = &neuron;
        }
    }

    if (bmu)
    {
        std::cout << "BMU para la entrada está en posición: (" << bmu->x << ", " << bmu->y << ")\n";
    }
}

void KohonenSOM::printMapActivity() const
{
    std::cout << "\nMapa de actividad (conteo de BMUs):\n";
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            std::cout << std::setw(3) << neurons[j * width + i].bmu_count << " ";
        }
        std::cout << std::endl;
    }
}
