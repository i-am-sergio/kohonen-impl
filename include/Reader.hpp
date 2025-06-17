#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class Reader {
public:
  static void load_csv(const std::string &filename,
                       std::vector<std::vector<double>> &X,
                       std::vector<std::vector<double>> &Y,
                       int num_classes,
                       bool header = false,
                       size_t max_rows = 0) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
      return;
    }

    std::string line;
    size_t row_count = 0;

    if (header) {
      std::getline(file, line);
    }

    while (std::getline(file, line)) {
      if (max_rows > 0 && row_count >= max_rows)
        break;

      std::stringstream ss(line);
      std::string token;
      std::vector<double> row;

      while (std::getline(ss, token, ',')) {
        try {
          row.push_back(std::stod(token));
        } catch (const std::invalid_argument &ia) {
          std::cerr << "Valor inválido en CSV: " << token << std::endl;
          continue;
        }
      }

      if (row.size() <= num_classes) {
        std::cerr << "Fila inválida con " << row.size()
                  << " columnas (esperado más de " << num_classes << ")." << std::endl;
        continue;
      }

      size_t image_size = row.size() - num_classes;

      std::vector<double> x_pixels(row.begin(), row.begin() + image_size);
      std::vector<double> y_labels(row.begin() + image_size, row.end());

      X.push_back(x_pixels);
      Y.push_back(y_labels);

      row_count++;
    }

    file.close();
  }
};
