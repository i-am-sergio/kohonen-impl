#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

std::vector<std::vector<double>> load_som_weights(const std::string& filename) {
    std::vector<std::vector<double>> all_weights;
    std::ifstream in(filename, std::ios::binary);
    
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading\n";
        return all_weights;
    }

    // Leer dimensiones
    size_t num_neurons, weights_dim;
    in.read(reinterpret_cast<char*>(&num_neurons), sizeof(num_neurons));
    in.read(reinterpret_cast<char*>(&weights_dim), sizeof(weights_dim));

    if (num_neurons == 0 || weights_dim == 0) {
        std::cerr << "Error: Invalid file format or empty data\n";
        return all_weights;
    }

    // Reservar espacio
    all_weights.resize(num_neurons, std::vector<double>(weights_dim));

    // Leer todos los pesos
    for (size_t i = 0; i < num_neurons; ++i) {
        in.read(reinterpret_cast<char*>(all_weights[i].data()), weights_dim * sizeof(double));
        
        // Verificar si la lectura fue exitosa
        if (in.gcount() != static_cast<std::streamsize>(weights_dim * sizeof(double))) {
            std::cerr << "Error: Failed to read complete weight data for neuron " << i << "\n";
            return {};
        }
    }

    in.close();
    return all_weights;
}

std::vector<std::vector<double>> load_som_weights_txt(const std::string& filename) {
    std::vector<std::vector<double>> weights;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> neuron_weights;
        std::stringstream ss(line);
        std::string value_str;
        
        while (std::getline(ss, value_str, ',')) {
            // Eliminar espacios en blanco
            value_str.erase(0, value_str.find_first_not_of(" \t"));
            value_str.erase(value_str.find_last_not_of(" \t") + 1);
            
            if (!value_str.empty()) {
                try {
                    // Configurar el locale para stod
                    char* end;
                    double weight = std::strtod(value_str.c_str(), &end);
                    
                    // Verificar que se procesó todo el string
                    if (end != value_str.c_str() + value_str.size()) {
                        throw std::invalid_argument("caracteres adicionales");
                    }
                    
                    neuron_weights.push_back(weight);
                } catch (...) {
                    throw std::runtime_error("Error al convertir valor: '" + value_str + "'");
                }
            }
        }
        
        if (!neuron_weights.empty()) {
            weights.push_back(neuron_weights);
        }
    }

    if (weights.empty()) {
        throw std::runtime_error("Archivo vacío o sin datos válidos");
    }

    // Verificación de consistencia...
    return weights;
}