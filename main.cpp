#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include "Reader.hpp"
#include "som.hpp"
#include "visualizer.hpp"

int main(int argc, char **argv)
{
    // Carga de datos
    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> Y_train;

    Reader::load_csv("database/mnist_train_flat_3.csv", X_train, Y_train, 60000);
    static std::vector<int> ys;
    ys.clear();
    for (const auto &output_row : Y_train)
    {
        int max_index = std::distance(output_row.begin(), std::max_element(output_row.begin(), output_row.end()));
        ys.push_back(max_index);
    }

    // Dividir sin barajar (primer 20% para validación)
    size_t total = X_train.size();
    size_t val_size = total / 5; // 20%
    size_t train_size = total - val_size;

    // Crear subconjuntos sin copiar innecesariamente
    std::vector<std::vector<double>> X_val(X_train.begin(), X_train.begin() + val_size);
    std::vector<std::vector<double>> Y_val(Y_train.begin(), Y_train.begin() + val_size);
    std::vector<int> Y_val_labels(ys.begin(), ys.begin() + val_size);

    X_train.erase(X_train.begin(), X_train.begin() + val_size);
    Y_train.erase(Y_train.begin(), Y_train.begin() + val_size);
    ys.erase(ys.begin(), ys.begin() + val_size);

    cout << "Nro de X de entrenamiento: " << X_train.size() << endl;
    cout << "Nro de Y de entrenamiento: " << ys.size() << endl;
    cout << "Nro de X de validación: " << X_val.size() << endl;
    cout << "Nro de Y de validación: " << Y_val_labels.size() << endl;

    // Test
    std::vector<std::vector<double>> X_test;
    std::vector<std::vector<double>> Y_test;
    Reader::load_csv("database/mnist_test_flat.csv", X_test, Y_test, 10000);
    std::vector<int> Y_test_labels;
    Y_test_labels.clear();
    for (const auto &output_row : Y_test)
    {
        int max_index = std::distance(output_row.begin(), std::max_element(output_row.begin(), output_row.end()));
        Y_test_labels.push_back(max_index);
    }
    cout << "Nro de X de test: " << X_test.size() << endl;
    cout << "Nro de Y de test: " << Y_test_labels.size() << endl;

    if (X_train.empty())
    {
        cerr << "Error: No se encontraron datos de entrenamiento.\n";
        return 1;
    }
    SOM som(784);
    som.train_and_evaluate(X_train, X_val, Y_val_labels, X_test, Y_test_labels, 10);
    Visualizer::show(som.get_neurons(), argc, argv);
    return 0;
}