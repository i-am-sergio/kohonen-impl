#include <algorithm>
#include <iostream>
#include <vector>

#include "Reader.hpp"
#include "RedKohonen.hpp"

using namespace std;

// Función para convertir etiquetas de one-hot a entero
vector<int> onehot_to_labels(const vector<vector<double>> &Y_onehot)
{
  vector<int> labels;
  labels.reserve(Y_onehot.size());
  for (const auto &row : Y_onehot)
  {
    auto max_it = max_element(row.begin(), row.end());
    labels.push_back(distance(row.begin(), max_it));
  }
  return labels;
}

int main(int argc, char **argv)
{
  // --- PARÁMETROS CONFIGURABLES ---
  const int DIM_X = 10;
  const int DIM_Y = 10;
  const int DIM_Z = 10;
  const int EPOCHS = 5;
  const double LEARNING_RATE = 0.5;
  const int INPUT_DIM = 784;            // 28x28 pixeles
  const double VALIDATION_SPLIT = 0.20; // 20% para validación
  const string WEIGHTS_FILENAME = "mnist_gaussian_radius";

  // --- 1. CARGA DE DATOS ---
  cout << "Cargando datos de entrenamiento..." << endl;
  vector<vector<double>> X_full, Y_full_onehot;
  Reader::load_csv("database/mnist_train_flat_3.csv", X_full, Y_full_onehot, 10);

  if (X_full.empty())
  {
    cerr << "Error: No se pudieron cargar los datos de entrenamiento." << endl;
    return 1;
  }

  cout << "\nCargando datos de prueba..." << endl;
  vector<vector<double>> X_test, Y_test_onehot;
  Reader::load_csv("database/mnist_test_flat.csv", X_test, Y_test_onehot, 10);
  if (X_test.empty())
  {
    cerr << "Error: No se pudieron cargar los datos de prueba." << endl;
    return 1;
  }

  // --- 2. DIVISIÓN DE DATOS (TRAIN/VALIDATION) ---
  size_t total_samples = X_full.size();
  size_t val_size = static_cast<size_t>(total_samples * VALIDATION_SPLIT);
  vector<vector<double>> X_val(X_full.begin(), X_full.begin() + val_size);
  vector<vector<double>> Y_val_onehot(Y_full_onehot.begin(), Y_full_onehot.begin() + val_size);
  vector<vector<double>> X_train(X_full.begin() + val_size, X_full.end());
  vector<int> Y_val = onehot_to_labels(Y_val_onehot);
  vector<int> Y_test = onehot_to_labels(Y_test_onehot);

  cout << "Total de muestras: " << total_samples << endl;
  cout << "Muestras de entrenamiento: " << X_train.size() << endl;
  cout << "Muestras de validacion: " << X_val.size() << endl;
  cout << "Muestras de prueba: " << X_test.size() << endl;

  RedKohonen som(INPUT_DIM, DIM_X, DIM_Y, DIM_Z, LEARNING_RATE, EPOCHS,
                 NeighborhoodMode::GAUSSIAN_RADIUS);
  cout << "\nIniciando entrenamiento de la red de Kohonen..." << endl;
  som.set_validation_data(X_val, Y_val);
  som.train_test(X_train, X_test, Y_test, WEIGHTS_FILENAME);
  return 0;
}

