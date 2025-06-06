#include "Reader.hpp"
#include "MNISTViewer.hpp" // Incluimos nuestra clase todo-en-uno
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    // Cargar datos de entrenamiento
    vector<vector<float>> raw_X_train;
    vector<vector<float>> raw_Y_train;

    // Carga TODAS las imágenes de entrenamiento (60000)
    // Asegúrate de que la ruta sea correcta para tu sistema
    Reader::load_csv("../topicos-inteligencia-artificial/lab4/MNISTdataset/mnist_test_flat.csv", raw_X_train, raw_Y_train, 60000);

    cout << "Nro de X de entrenamiento: " << raw_X_train.size() << endl;
    cout << "Nro de Y de entrenamiento: " << raw_Y_train.size() << endl;

    if (raw_X_train.empty()) {
        cerr << "No se cargaron datos de entrenamiento. Saliendo." << endl;
        return 1;
    }

    // Crear una instancia de la clase MNISTViewer, pasando TODAS las imágenes
    MNISTViewer viewer(raw_X_train, 28, 28);

    // Ejecutar el visualizador
    viewer.run(argc, argv);

    return 0;
}