#pragma once

#include "Neuron.hpp"
#include <tuple>
#include <vector>
#include <string>
#include <algorithm>

enum class NeighborhoodMode
{
  BMU_ONLY,        // Solo actualiza la BMU
  GAUSSIAN_RADIUS, // Vecinos con influencia gaussiana
  CONSTANT_RADIUS  // Vecinos con influencia constante dentro del radio
};

class RedKohonen
{
private:
  int dim_x, dim_y, dim_z;
  int input_dim;
  int total_neurons;
  double initial_learning_rate;
  int epochs;
  double time_constant;
  double initial_radius;

  std::vector<Neuron> neurons;
  std::vector<std::vector<double>> X_val_data;
  std::vector<int> Y_val_labels;

  bool validation_enabled = false;
  NeighborhoodMode mode = NeighborhoodMode::GAUSSIAN_RADIUS;

  int find_bmu(const std::vector<double> &input) const;

public:
  RedKohonen(int inputDim, int dX, int dY, int dZ, double initialLR = 0.0, int numEpochs = 0,
             NeighborhoodMode mode_ = NeighborhoodMode::GAUSSIAN_RADIUS)
      : input_dim(inputDim), dim_x(dX), dim_y(dY), dim_z(dZ),
        initial_learning_rate(initialLR), epochs(numEpochs), mode(mode_)
  {
    total_neurons = dim_x * dim_y * dim_z;
    neurons.reserve(total_neurons);
    for (int i = 0; i < total_neurons; ++i)
      neurons.emplace_back(initialLR > 0 ? Neuron(input_dim) : Neuron());

    if (numEpochs > 0)
    {
      initial_radius = std::max({dim_x, dim_y, dim_z}) / 2.0;
      time_constant = epochs / log(initial_radius);
    }
  }

  void assign_labels(const std::vector<std::vector<double>> &X_val, const std::vector<int> &Y_val);
  void set_validation_data(const std::vector<std::vector<double>> &X_val, const std::vector<int> &Y_val);
  int predict(const std::vector<double> &x) const;
  std::pair<int, std::tuple<int, int, int>> predict_with_coords(const std::vector<double> &x) const;
  std::tuple<int, int, int> find_bmu_coords(const std::vector<double> &input) const;
  void train(int epoch, const std::vector<std::vector<double>> &X_train, std::ofstream *log_file);
  float test_accuracy(const std::vector<std::vector<double>> &X_test, const std::vector<int> &Y_test) const;
  void train_test(const std::vector<std::vector<double>> &X_train,
                  const std::vector<std::vector<double>> &X_test,
                  const std::vector<int> &Y_test, const std::string &weights_filename = "base");
  void save_weights(const std::string &filename) const;
  void load_weights(const std::string &filename);

  const std::vector<Neuron> &get_neurons() const { return neurons; }
  int get_dim_x() const { return dim_x; }
  int get_dim_y() const { return dim_y; }
  int get_dim_z() const { return dim_z; }
};
