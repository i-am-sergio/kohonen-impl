#pragma once

#include "Neuron.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>
#include <vector>

class RedKohonen {
private:
  int dim_x, dim_y, dim_z;
  int input_dim;
  int total_neurons;
  double initial_learning_rate;
  int epochs;

  std::vector<Neuron> neurons;
  double time_constant;
  double initial_radius;

  std::vector<std::vector<double>> X_val_data;
  std::vector<int> Y_val_labels;
  bool validation_enabled = false;


public:
  RedKohonen(int inputDim, int dX, int dY, int dZ, double initialLR = 0.0, int numEpochs = 0)
      : input_dim(inputDim), dim_x(dX), dim_y(dY), dim_z(dZ), initial_learning_rate(initialLR), epochs(numEpochs) {

    total_neurons = dim_x * dim_y * dim_z;
    neurons.reserve(total_neurons);
    for (int i = 0; i < total_neurons; ++i) {
      // Cuando solo cargamos, no necesitamos inicializar con datos aleatorios
      if (initialLR > 0) { // Asumimos que si LR es > 0, es para entrenar
        neurons.emplace_back(input_dim);
      } else {
        // Constructor por defecto para cuando solo vamos a cargar pesos
        neurons.emplace_back();
      }
    }

    if (numEpochs > 0) {
      initial_radius = std::max({dim_x, dim_y, dim_z}) / 2.0;
      time_constant = epochs / log(initial_radius);
    }
  }

  // Devuelve el índice de la neurona con el peso más cercano al vector de entrada
  int find_bmu(const std::vector<double> &input) const {
    int bmu_idx = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < total_neurons; ++i) {
      double d = neurons[i].distance_sq(input);
      if (d < min_dist) {
        min_dist = d;
        bmu_idx = i;
      }
    }
    return bmu_idx;
  }
  void set_validation_data(const std::vector<std::vector<double>> &X_val, const std::vector<int> &Y_val) {
    X_val_data = X_val;
    Y_val_labels = Y_val;
    validation_enabled = true;
  }


  void train(const std::vector<std::vector<double>> &X_train) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
      auto start = start_timer();

      // Tasa de aprendizaje y radio de vecindad que decaen con el tiempo
      double current_lr = initial_learning_rate * exp(-(double)epoch / epochs);
      double current_radius = initial_radius * exp(-(double)epoch / time_constant);
      double radius_sq = current_radius * current_radius;

      std::cout << "Epoca " << epoch + 1 << "/" << epochs
                << " | LR: " << current_lr
                << " | Radio: " << current_radius << std::endl;

      int sample_count = 0;
      for (const auto &sample : X_train) {
        std::cout << "  Procesando muestra " << ++sample_count << "/" << X_train.size() << "\r";

        int bmu_idx = find_bmu(sample);
        int bmu_z = bmu_idx / (dim_x * dim_y);
        int bmu_y = (bmu_idx % (dim_x * dim_y)) / dim_x;
        int bmu_x = bmu_idx % dim_x;

        #pragma omp parallel for
        for (int i = 0; i < total_neurons; ++i) {
          int z = i / (dim_x * dim_y);
          int y = (i % (dim_x * dim_y)) / dim_x;
          int x = i % dim_x;

          double dist_to_bmu_sq = pow(x - bmu_x, 2) + pow(y - bmu_y, 2) + pow(z - bmu_z, 2);

          if (dist_to_bmu_sq < radius_sq) {
            // Función de influencia Gaussiana
            double influence = exp(-dist_to_bmu_sq / (2 * radius_sq));
            neurons[i].update_weights(sample, current_lr, influence);
          }
        }
      }

      std::cout << std::endl;
      double duration = stop_timer(start);
      print_duration(duration, "Tiempo de entrenamiento");

      // Evaluación en validación (si está disponible)
      if (validation_enabled) {
        float val_acc = test_accuracy(X_val_data, Y_val_labels);
        std::cout << "  >> Precision en validacion: " << val_acc * 100.0f << "%" << std::endl;
      }
    }
  }


  // Asigna una etiqueta a cada neurona basada en los datos de validación
  void assign_labels(const std::vector<std::vector<double>> &X_val, const std::vector<int> &Y_val) {
    std::cout << "Asignando etiquetas a las neuronas..." << std::endl;
    std::vector<std::vector<int>> hits(total_neurons);

    for (size_t i = 0; i < X_val.size(); ++i) {
      int bmu_idx = find_bmu(X_val[i]);
      hits[bmu_idx].push_back(Y_val[i]);
    }

#pragma omp parallel for
    for (int i = 0; i < total_neurons; ++i) {
      if (!hits[i].empty()) {
        std::vector<int> counts(10, 0); // Contadores para los dígitos 0-9
        for (int label : hits[i]) {
          counts[label]++;
        }
        auto max_it = std::max_element(counts.begin(), counts.end());
        int majority_label = std::distance(counts.begin(), max_it);
        neurons[i].set_label(majority_label);
      }
    }
    std::cout << "Etiquetado completado." << std::endl;
  }

  int predict(const std::vector<double> &x) const {
    int bmu_idx = find_bmu(x);
    return neurons[bmu_idx].get_label();
  }

  float test_accuracy(const std::vector<std::vector<double>> &X_test, const std::vector<int> &Y_test) const {
    int correct_predictions = 0;
#pragma omp parallel for reduction(+ : correct_predictions)
    for (size_t i = 0; i < X_test.size(); ++i) {
      if (predict(X_test[i]) == Y_test[i]) {
        correct_predictions++;
      }
    }
    return static_cast<float>(correct_predictions) / X_test.size();
  }

  const std::vector<Neuron> &get_neurons() const { return neurons; }

  void save_weights(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: No se pudo abrir el archivo para guardar pesos: " << filename << std::endl;
      return;
    }

    // Guardar las dimensiones de la red primero
    // file << dim_x << " " << dim_y << " " << dim_z << std::endl;

    for (const auto &neuron : neurons) {
      const auto &weights = neuron.get_weights();
      for (size_t i = 0; i < weights.size(); ++i) {
        file << weights[i] << (i == weights.size() - 1 ? "" : ",");
      }
      file << std::endl;
    }
    file.close();
    std::cout << "Pesos guardados en " << filename << std::endl;
  }

  void load_weights(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: No se pudo abrir el archivo para cargar pesos: " << filename << std::endl;
      return;
    }

    std::string line;
    // Leer las dimensiones de la red
    if (std::getline(file, line)) {
      std::stringstream ss(line);
      ss >> dim_x >> dim_y >> dim_z;
      total_neurons = dim_x * dim_y * dim_z;
      neurons.clear();
      neurons.reserve(total_neurons);
    } else {
      std::cerr << "Error: Archivo de pesos vacío o corrupto." << std::endl;
      return;
    }

    // Leer los pesos de cada neurona
    while (std::getline(file, line)) {
      // Ignorar líneas vacías o que solo contienen espacios en blanco
      if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace)) {
        continue;
      }

      std::vector<double> weights;
      weights.reserve(784); // Pequeña optimización
      std::stringstream ss(line);
      std::string value;
      while (std::getline(ss, value, ',')) {
        try {
          weights.push_back(std::stod(value));
        } catch (const std::invalid_argument &e) {
          std::cerr << "Advertencia: Valor no numérico encontrado: '" << value << "'. Ignorando." << std::endl;
        } catch (const std::out_of_range &e) {
          // std::cerr << "Advertencia: Valor fuera de rango encontrado: '" << value << "'. Se usará 0." << std::endl;
          weights.push_back(0.0);
        }
      }

      if (weights.empty())
        continue; // Si la línea no produjo pesos, la ignoramos.

      Neuron n;
      n.set_weights_from_load(weights);
      neurons.push_back(n);
    }

    if (neurons.size() != static_cast<size_t>(total_neurons)) {
      std::cerr << "Advertencia: El número de neuronas cargadas (" << neurons.size()
                << ") no coincide con las dimensiones del archivo (" << total_neurons << ")." << std::endl;
    }

    file.close();
    std::cout << "Pesos cargados desde " << filename << std::endl;
    std::cout << "Dimensiones de la red cargada: " << dim_x << "x" << dim_y << "x" << dim_z << std::endl;
  }
  // getters para dimensiones
  int get_dim_x() const { return dim_x; }
  int get_dim_y() const { return dim_y; }
  int get_dim_z() const { return dim_z; }
};
