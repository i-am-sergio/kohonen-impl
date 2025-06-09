#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <algorithm> // Para std::max
#include "Reader.hpp"

using namespace std;

struct Neuron {
    vector<float> weights;
    int x, y;
    int bmu_count = 0;

    Neuron(int dim, int xpos, int ypos) : x(xpos), y(ypos) {
        weights.resize(dim);
        for (auto& w : weights)
            w = static_cast<float>(rand()) / RAND_MAX;
    }
};

float euclideanDistance(const vector<float>& a, const vector<float>& b) {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        dist += pow(a[i] - b[i], 2);
    return sqrt(dist);
}

float neighborhoodFunction(int dx, int dy, float radius) {
    float dist2 = static_cast<float>(dx * dx + dy * dy);
    return exp(-dist2 / (2 * radius * radius));
}

void updateWeights(Neuron& neuron, const vector<float>& input, float lr, float influence) {
    for (size_t i = 0; i < neuron.weights.size(); ++i)
        neuron.weights[i] += lr * influence * (input[i] - neuron.weights[i]);
}

Neuron* findBMU(vector<Neuron>& neurons, const vector<float>& input) {
    Neuron* bmu = &neurons[0];
    float minDist = euclideanDistance(input, neurons[0].weights);

    for (auto& neuron : neurons) {
        float dist = euclideanDistance(input, neuron.weights);
        if (dist < minDist) {
            minDist = dist;
            bmu = &neuron;
        }
    }
    bmu->bmu_count++;
    return bmu;
}

void printBMUPosition(const vector<Neuron>& neurons, const vector<float>& input) {
    const Neuron* bmu = nullptr;
    float minDist = 1e9f;
    for (const auto& neuron : neurons) {
        float dist = euclideanDistance(input, neuron.weights);
        if (dist < minDist) {
            minDist = dist;
            bmu = &neuron;
        }
    }

    if (bmu) {
        cout << "BMU para la entrada está en posición: (" << bmu->x << ", " << bmu->y << ")\n";
    }
}

void printMapActivity(const vector<Neuron>& neurons, int width, int height) {
    cout << "\nMapa de actividad (conteo de BMUs):\n";
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            cout << setw(3) << neurons[j * width + i].bmu_count << " ";
        }
        cout << endl;
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    // Carga de datos
    vector<vector<float>> raw_X_train;
    vector<vector<float>> raw_Y_train;

    Reader::load_csv("../topicos-inteligencia-artificial/datasets/MNISTdataset/mnist_train_flat.csv", raw_X_train, raw_Y_train, 60000);

    cout << "Nro de X de entrenamiento: " << raw_X_train.size() << endl;
    cout << "Nro de Y de entrenamiento: " << raw_Y_train.size() << endl;

    if (raw_X_train.empty()) {
        cerr << "Error: No se encontraron datos de entrenamiento.\n";
        return 1;
    }

    int inputDim = 784;
    int width = 10, height = 10, epochs = 10; // Epochs reducido para pruebas
    float learningRate = 0.1f;
    float radius = max(width, height) / 2.0f;

    vector<Neuron> neurons;
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < height; ++j)
            neurons.emplace_back(inputDim, i, j);

    // Entrenamiento del SOM
    for (int epoch = 0; epoch < epochs; ++epoch) {
        cout << "Epoch " << epoch + 1 << "/" << epochs << endl;
        for (const auto& input : raw_X_train) {
            Neuron* bmu = findBMU(neurons, input);
            for (auto& neuron : neurons) {
                int dx = neuron.x - bmu->x;
                int dy = neuron.y - bmu->y;
                float influence = neighborhoodFunction(dx, dy, radius);
                updateWeights(neuron, input, learningRate, influence);
            }
        }
        learningRate *= 0.99f;
        radius *= 0.99f;
    }

    cout << "Entrenamiento completado.\n";

    // Probar predicción con una nueva entrada (reutilizamos una del dataset)
    printBMUPosition(neurons, raw_X_train[0]);

    // Mostrar mapa de actividad
    printMapActivity(neurons, width, height);

    return 0;
}
