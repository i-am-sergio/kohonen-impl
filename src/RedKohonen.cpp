#include "RedKohonen.hpp"
#include "Utils.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <omp.h>
#include <cctype>
#include <filesystem>
#include <iomanip>

void RedKohonen::set_validation_data(const std::vector<std::vector<double>> &X_val, const std::vector<int> &Y_val)
{
    X_val_data = X_val;
    Y_val_labels = Y_val;
    validation_enabled = true;
}

int RedKohonen::predict(const std::vector<double> &x) const
{
    int bmu_idx = find_bmu(x);
    return neurons[bmu_idx].get_label();
}

std::pair<int, std::tuple<int, int, int>> RedKohonen::predict_with_coords(const std::vector<double> &x) const
{
    int idx = find_bmu(x);
    int z = idx / (dim_x * dim_y);
    int y = (idx % (dim_x * dim_y)) / dim_x;
    int x_ = idx % dim_x;
    return {neurons[idx].get_label(), {x_, y, z}};
}

std::tuple<int, int, int> RedKohonen::find_bmu_coords(const std::vector<double> &input) const
{
    int idx = find_bmu(input);
    int z = idx / (dim_x * dim_y);
    int y = (idx % (dim_x * dim_y)) / dim_x;
    int x = idx % dim_x;
    return {x, y, z};
}

int RedKohonen::find_bmu(const std::vector<double> &input) const
{
    int bmu_idx = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < total_neurons; ++i)
    {
        double d = neurons[i].distance_sq(input);
        if (d < min_dist)
        {
            min_dist = d;
            bmu_idx = i;
        }
    }
    return bmu_idx;
}

void RedKohonen::assign_labels(const std::vector<std::vector<double>> &X_val, const std::vector<int> &Y_val)
{
    std::vector<std::vector<int>> hits(total_neurons);
    for (size_t i = 0; i < X_val.size(); ++i)
    {
        int bmu_idx = find_bmu(X_val[i]);
        hits[bmu_idx].push_back(Y_val[i]);
    }

#pragma omp parallel for
    for (int i = 0; i < total_neurons; ++i)
    {
        if (!hits[i].empty())
        {
            std::vector<int> counts(10, 0);
            for (int label : hits[i])
            {
                counts[label]++;
            }
            int majority_label = std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));
            neurons[i].set_label(majority_label);
        }
    }
}

void RedKohonen::train(int epoch, const std::vector<std::vector<double>> &X_train, std::ofstream *log_file)
{
    auto start = start_timer();

    double current_lr = initial_learning_rate;
    double current_radius = initial_radius;
    double radius_sq = current_radius * current_radius;

    if (mode == NeighborhoodMode::GAUSSIAN_RADIUS || mode == NeighborhoodMode::CONSTANT_RADIUS)
    {
        current_lr = initial_learning_rate * exp(-(double)epoch / epochs);
        current_radius = initial_radius * exp(-(double)epoch / time_constant);
        radius_sq = current_radius * current_radius;
    }
    int sample_count = 0;
    for (const auto &sample : X_train)
    {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ": " << ++sample_count << "/" << X_train.size() << "\r";
        std::cout.flush();

        int bmu_idx = find_bmu(sample);
        int bmu_z = bmu_idx / (dim_x * dim_y);
        int bmu_y = (bmu_idx % (dim_x * dim_y)) / dim_x;
        int bmu_x = bmu_idx % dim_x;

#pragma omp parallel for
        for (int i = 0; i < total_neurons; ++i)
        {
            int z = i / (dim_x * dim_y);
            int y = (i % (dim_x * dim_y)) / dim_x;
            int x = i % dim_x;

            double dist_to_bmu_sq = pow(x - bmu_x, 2) + pow(y - bmu_y, 2) + pow(z - bmu_z, 2);
            switch (mode)
            {
            case NeighborhoodMode::BMU_ONLY:
                if (i == bmu_idx)
                    neurons[i].update_weights(sample, current_lr, 1.0);
                break;

            case NeighborhoodMode::GAUSSIAN_RADIUS:
                if (dist_to_bmu_sq < radius_sq)
                {
                    double influence = exp(-dist_to_bmu_sq / (2 * radius_sq));
                    neurons[i].update_weights(sample, current_lr, influence);
                }
                break;

            case NeighborhoodMode::CONSTANT_RADIUS:
                if (dist_to_bmu_sq < radius_sq)
                    neurons[i].update_weights(sample, current_lr, 1.0);
                break;
            }
        }
    }

    double duration = stop_timer(start);
    float val_acc = 0.0f;

    std::cout << "Epoch " << epoch + 1 << "/" << epochs
              << " | lr: " << current_lr;

    if (mode != NeighborhoodMode::BMU_ONLY)
        std::cout << " | Radius: " << current_radius;

    std::cout << " | Train Time: " << duration << "s";

    if (validation_enabled)
    {
        assign_labels(X_val_data, Y_val_labels);
        val_acc = test_accuracy(X_val_data, Y_val_labels);
        std::cout << " | Val Acc: " << val_acc * 100.0f << "%";
    }

    if (log_file)
    {
        (*log_file) << "Epoch " << epoch + 1 << "/" << epochs
                    << " | lr: " << current_lr;

        if (mode != NeighborhoodMode::BMU_ONLY)
            (*log_file) << " | Radius: " << current_radius;

        (*log_file) << " | Train Time: " << duration << "s";

        if (validation_enabled)
            (*log_file) << " | Val Acc: " << val_acc * 100.0f << "%";
    }
}

float RedKohonen::test_accuracy(const std::vector<std::vector<double>> &X_test, const std::vector<int> &Y_test) const
{
    int correct_predictions = 0;
#pragma omp parallel for reduction(+ : correct_predictions)
    for (size_t i = 0; i < X_test.size(); ++i)
    {
        if (predict(X_test[i]) == Y_test[i])
        {
            correct_predictions++;
        }
    }
    return static_cast<float>(correct_predictions) / X_test.size();
}

void RedKohonen::train_test(const std::vector<std::vector<double>> &X_train,
                            const std::vector<std::vector<double>> &X_test,
                            const std::vector<int> &Y_test, const std::string &weights_filename)
{
    std::string output_dir = "output/" + weights_filename;
    std::filesystem::create_directories(output_dir);
    std::ofstream log_file(output_dir + "/log.txt");

    float best_test_acc = 0.0f;
    int best_epoch = -1;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto start = start_timer();
        train(epoch, X_train, &log_file);
        float test_acc = test_accuracy(X_test, Y_test);
        double total_time = stop_timer(start);
        std::cout << " | Test Acc: " << test_acc * 100.0f
                  << "% | Total Time: " << total_time << "s" << std::endl;

        if (log_file.is_open())
        {
            log_file << " | Test Acc: " << test_acc * 100.0f
                     << "% | Total Time: " << total_time << "s" << std::endl;
        }

        if ((epoch + 1) % 5 == 0)
        {
            save_weights(output_dir + "/checkpoint.dat");
        }

        // Guardar el mejor modelo
        if (test_acc > best_test_acc)
        {
            best_test_acc = test_acc;
            best_epoch = epoch;
            save_weights(output_dir + "/best_model.dat");
        }
    }
    save_weights(output_dir + "/final.dat");
    if (log_file.is_open())
    {
        log_file << "Best Test Accuracy: " << best_test_acc * 100.0f
                 << "% at epoch " << (best_epoch + 1) << std::endl;
    }
    std::cout << "Best Test Accuracy: " << best_test_acc * 100.0f
              << "% at epoch " << (best_epoch + 1) << std::endl;
    log_file.close();
}

void RedKohonen::save_weights(const std::string &filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error al guardar pesos: " << filename << std::endl;
        return;
    }

    for (const auto &neuron : neurons)
    {
        const auto &weights = neuron.get_weights();
        for (size_t i = 0; i < weights.size(); ++i)
            file << weights[i] << (i == weights.size() - 1 ? "" : ",");
        file << std::endl;
    }
    file.close();
}

void RedKohonen::load_weights(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error al abrir archivo de pesos: " << filename << std::endl;
        return;
    }

    std::string line;
    if (std::getline(file, line))
    {
        std::stringstream ss(line);
        ss >> dim_x >> dim_y >> dim_z;
        total_neurons = dim_x * dim_y * dim_z;
        neurons.clear();
        neurons.reserve(total_neurons);
    }
    else
    {
        std::cerr << "Error: Archivo de pesos vacío o corrupto." << std::endl;
        return;
    }

    while (std::getline(file, line))
    {
        if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace))
            continue;

        std::vector<double> weights;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ','))
        {
            try
            {
                weights.push_back(std::stod(value));
            }
            catch (...)
            {
                weights.push_back(0.0);
            }
        }

        if (!weights.empty())
        {
            Neuron n;
            n.set_weights_from_load(weights);
            neurons.push_back(n);
        }
    }

    if (neurons.size() != static_cast<size_t>(total_neurons))
    {
        std::cerr << "Advertencia: número de neuronas cargadas no coincide con el esperado." << std::endl;
    }

    file.close();
    std::cout << "Pesos cargados desde " << filename << std::endl;
}