#pragma once

#include <random>
#include <vector>

class Neuron {
private:
  std::vector<double> weights;
  int label = -1; // -1 indica que no está etiquetada

public:
  Neuron() = default;

  Neuron(int n_inputs) : weights(n_inputs) {
    // Inicialización aleatoria de pesos entre 0 y 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < n_inputs; ++i) {
      weights[i] = dis(gen);
    }
  }

  // Distancia euclidiana al cuadrado (más eficiente)
  double distance_sq(const std::vector<double> &input) const {
    double d = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
      double diff = input[i] - weights[i];
      d += diff * diff;
    }
    return d;
  }

  void update_weights(const std::vector<double> &input, double learning_rate, double influence) {
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] += learning_rate * influence * (input[i] - weights[i]);
    }
  }

  void set_label(int lbl) { label = lbl; }
  int get_label() const { return label; }
  const std::vector<double> &get_weights() const { return weights; }

  void set_weights_from_load(const std::vector<double> &new_weights) { weights = new_weights; }
};
