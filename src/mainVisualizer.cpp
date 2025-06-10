#include "RedKohonen.hpp"
#include "Visualizer.hpp"
#include <iostream>

int main(int argc, char **argv) {
  const std::string WEIGHTS_FILENAME = "som_weights.txt";

  // Creamos una instancia de la red sin parámetros de entrenamiento
  // Las dimensiones se leerán del archivo de pesos.
  RedKohonen som(0, 0, 0, 0);

  std::cout << "Cargando pesos desde " << WEIGHTS_FILENAME << "..." << std::endl;
  som.load_weights(WEIGHTS_FILENAME);

  if (som.get_neurons().empty()) {
    std::cerr << "No se cargaron neuronas. Terminando programa." << std::endl;
    return 1;
  }

  std::cout << "Iniciando visualizador..." << std::endl;
  Visualizer::show(som.get_neurons(), som.get_dim_x(), som.get_dim_y(), som.get_dim_z(), argc, argv);

  return 0;
}
