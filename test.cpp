#include "Reader.hpp"
#include <iostream> // Make sure iostream is included for cout
#include <vector>   // Make sure vector is included for std::vector

int main() {
    // Load training data
    std::vector<std::vector<float>> raw_X_train;
    std::vector<std::vector<float>> raw_Y_train;

    // Adjust the path as necessary based on your project structure
    Reader::load_csv("../topicos-inteligencia-artificial/lab4/MNISTdataset/mnist_train_flat.csv", raw_X_train, raw_Y_train, 60000);

    std::cout << "Nro de X de entrenamiento: " << raw_X_train.size() << std::endl;
    std::cout << "Nro de Y de entrenamiento: " << raw_Y_train.size() << std::endl; // Corrected to raw_Y_train.size()

    // Select the first image
    if (raw_X_train.empty()) {
        std::cout << "No training data loaded. Exiting." << std::endl;
        return 1;
    }
    int c=1;
    while(c++){
        
        auto img1 = raw_X_train[c-2];
        
        const float threshold = 0.5f; // Common threshold for normalized pixel values
        
        std::cout << "\nPrimera imagen (binarizada 28x28):" << std::endl;
        for (int i = 0; i < 28; ++i) { // Iterate through rows
            for (int j = 0; j < 28; ++j) { // Iterate through columns
                // Calculate the index in the flat array (i * 28 + j)
                float pixel_value = img1[i * 28 + j];
                if (pixel_value > threshold) {
                    std::cout << "1 ";
            } else {
                std::cout << "0 ";
            }
        }
        std::cout << std::endl; // New line after each row
    }
    if(c==10) break;
    
}
    return 0;
}