#include <iostream>
#include "reader.hpp"
#include "kohonen.hpp"

int main()
{
    srand(static_cast<unsigned>(time(0)));

    std::vector<std::vector<float>> X_train, Y_train;
    std::vector<std::vector<float>> X_test, Y_test;
    Reader::load_csv("./database/mnist_train_flat_3.csv", X_train, Y_train, 60000);
    Reader::load_csv("./database/mnist_test_flat.csv", X_test, Y_test, 10000);

    if (X_train.empty())
    {
        std::cerr << "Error: No se encontraron datos de entrenamiento.\n";
        return 1;
    }

    KohonenSOM som(784, 10, 10, 10, 0.1f);
    som.train(X_train, X_test);
    std::cout << "Entrenamiento completado.\n";
    som.printBMUPosition(X_train[0]);
    som.printMapActivity();

    return 0;
}
