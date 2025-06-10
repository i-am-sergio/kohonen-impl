#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class Reader {
public:
  static void load_csv(const std::string &filename, std::vector<std::vector<double>> &X, std::vector<std::vector<double>> &Y,
                       size_t max_rows = 0) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: No se pudo abrir el archivo " << filename << std::endl;
      return;
    }

    std::string line;
    size_t row_count = 0;

    // Omitir la cabecera
    // getline(file, line);

    while (getline(file, line)) {
      if (max_rows > 0 && row_count >= max_rows)
        break;

      std::stringstream ss(line);
      std::string token;
      std::vector<double> row;
      row.reserve(794); // Pre-reservar memoria

      while (getline(ss, token, ',')) {
        try {
          row.push_back(stod(token));
        } catch (const std::invalid_argument &ia) {
          std::cerr << "Valor inválido en CSV: " << token << std::endl;
          continue;
        }
      }

      if (row.size() != 784 + 10) {
        std::cerr << "Fila inválida con " << row.size() << " columnas." << std::endl;
        continue;
      }

      // Normalizar pixeles a [0, 1] si no lo están ya
      std::vector<double> x_pixels(784);
      for (size_t i = 0; i < 784; ++i) {
        x_pixels[i] = row[i] / 255.0; // MNIST está en [0, 255]
      }

      X.push_back(x_pixels);
      Y.push_back({row.begin() + 784, row.end()});

      row_count++;
    }

    file.close();
  }
};
