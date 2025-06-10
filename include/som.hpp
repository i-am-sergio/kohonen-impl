#pragma once
#include "neuron.hpp"
#include <algorithm>
#include <omp.h>
#include <ctime>

class SOM
{
private:
    int dim_x = 10, dim_y = 10, dim_z = 10;
    int input_dim;
    double learning_rate = 0.1;
    std::vector<Neuron> neurons;

public:
    SOM(int input_dim) : input_dim(input_dim)
    {
        neurons.reserve(dim_x * dim_y * dim_z);
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (input_dim + 1));
        std::uniform_real_distribution<> dis(-limit, limit);
        for (int i = 0; i < dim_x * dim_y * dim_z; ++i)
            neurons.emplace_back(input_dim, gen, dis);
    }

    int find_bmu(const std::vector<double> &input) const
    {
        int bmu = 0;
        double min_dist = neurons[0].distance(input);

#pragma omp parallel
        {
            int local_bmu = -1;
            double local_min = std::numeric_limits<double>::max();

#pragma omp for nowait
            for (int i = 0; i < neurons.size(); ++i)
            {
                double d = neurons[i].distance(input);
                if (d < local_min)
                {
                    local_min = d;
                    local_bmu = i;
                }
            }

#pragma omp critical
            {
                if (local_min < min_dist)
                {
                    min_dist = local_min;
                    bmu = local_bmu;
                }
            }
        }

        return bmu;
    }

    void train_epoch(const std::vector<std::vector<double>> &X)
    {
        int i = 1;
        for (const auto &x : X)
        {
            std::cout << "Training on sample " << i++ << "/" << X.size() << "\r";
            int bmu = find_bmu(x);
            neurons[bmu].update(x, learning_rate);
        }
    }

    void validate(const std::vector<std::vector<double>> &X_val,
                  const std::vector<int> &Y_val)
    {
        static std::vector<std::vector<int>> hits(neurons.size());
#pragma omp parallel for
        for (auto &vec : hits)
            vec.clear();

#pragma omp parallel for
        for (int i = 0; i < X_val.size(); ++i)
        {
            int bmu = find_bmu(X_val[i]);
#pragma omp critical
            hits[bmu].push_back(Y_val[i]);
        }

#pragma omp parallel for
        for (int i = 0; i < neurons.size(); ++i)
        {
            if (!hits[i].empty())
            {
                std::vector<int> count(10, 0);
                for (int label : hits[i])
                    ++count[label];

                int majority_label = std::distance(count.begin(),
                                                   std::max_element(count.begin(), count.end()));
                neurons[i].set_label(majority_label);
            }
        }
    }

    int predict(const std::vector<double> &x) const
    {
        int bmu = find_bmu(x);
        return neurons[bmu].get_label();
    }

    float test(const std::vector<std::vector<double>> &X_test,
               const std::vector<int> &Y_test) const
    {
        int correct = 0;

#pragma omp parallel for reduction(+ : correct)
        for (int i = 0; i < X_test.size(); ++i)
        {
            int pred = predict(X_test[i]);
            if (pred == Y_test[i])
                correct++;
        }

        return correct * 1.0f / X_test.size();
    }

    const std::vector<Neuron> &get_neurons() const
    {
        return neurons;
    }

    float accuracy(const std::vector<std::vector<double>> &X,
                   const std::vector<int> &Y) const
    {
        int correct = 0;

#pragma omp parallel for reduction(+ : correct)
        for (int i = 0; i < X.size(); ++i)
        {
            int pred = predict(X[i]);
            if (pred == Y[i])
                ++correct;
        }

        return static_cast<float>(correct) / Y.size();
    }

    void train_and_evaluate(const std::vector<std::vector<double>> &X_train,
                            const std::vector<std::vector<double>> &X_val,
                            const std::vector<int> &Y_val,
                            const std::vector<std::vector<double>> &X_test,
                            const std::vector<int> &Y_test,
                            int epochs)
    {
        for (int e = 0; e < epochs; ++e)
        {
            clock_t start_train = clock();
            train_epoch(X_train);
            clock_t end_train = clock();
            double train_time = double(end_train - start_train) / CLOCKS_PER_SEC;

            clock_t start_val = clock();
            validate(X_val, Y_val);
            float val_acc = accuracy(X_val, Y_val);
            clock_t end_val = clock();
            double val_time = double(end_val - start_val) / CLOCKS_PER_SEC;

            clock_t start_test = clock();
            float test_acc = test(X_test, Y_test);
            clock_t end_test = clock();
            double test_time = double(end_test - start_test) / CLOCKS_PER_SEC;

            std::cout << "Epoch " << (e + 1)
                      << " | Train Time: " << train_time << "s"
                      << " | Val Acc: " << val_acc * 100 << "%"
                      << " | Val Time: " << val_time << "s"
                      << " | Test Acc: " << test_acc * 100 << "%"
                      << " | Test Time: " << test_time << "s\n";
        }
    }
};
