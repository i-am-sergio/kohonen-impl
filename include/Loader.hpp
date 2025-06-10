#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

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