#pragma once
#include <vector>
#include <random>
#include <cmath>

class Neuron
{
private:
    std::vector<double> weights;
    double bias;
    int label = -1;

public:
    Neuron() = default;

    template <typename RNG, typename Dist>
    Neuron(int n_inputs, RNG &gen, Dist &dis)
        : weights(n_inputs)
    {
        for (int i = 0; i < n_inputs; ++i)
            weights[i] = dis(gen);
        bias = 0.0;
    }

    double distance(const std::vector<double> &input) const
    {
        double d = 0.0;
        for (size_t i = 0; i < input.size(); ++i)
        {
            double diff = input[i] - weights[i];
            d += diff * diff;
        }
        return d;
    }

    void update(const std::vector<double> &input, double lr)
    {
        for (int i = 0; i < input.size(); ++i)
            weights[i] += lr * (input[i] - weights[i]);
    }

    void set_label(int lbl)
    {
        label = lbl;
    }

    int get_label() const
    {
        return label;
    }

    const std::vector<double> &get_weights() const
    {
        return weights;
    }
};
