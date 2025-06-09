#pragma once
#include <vector>

class Neuron
{
public:
    std::vector<float> weights;
    int x, y;
    int bmu_count;
    Neuron() = default;

    template <typename RNG, typename Dist>
    Neuron(int dim, int xpos, int ypos, RNG &gen, Dist &dis)
        : x(xpos), y(ypos), bmu_count(0), weights(dim)
    {
        for (int i = 0; i < dim; ++i)
            weights[i] = dis(gen);
    }
};