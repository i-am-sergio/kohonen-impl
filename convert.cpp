#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <iomanip>

using namespace std;

void convert_mnist_to_csv(const string &image_file, const string &label_file, const string &output_csv, int limit = -1)
{
    ifstream images(image_file, ios::binary);
    ifstream labels(label_file, ios::binary);
    ofstream output(output_csv);

    assert(images.is_open() && "No se pudo abrir el archivo de imagenes");
    assert(labels.is_open() && "No se pudo abrir el archivo de etiquetas");

    // Leer cabecera de imagenes
    int32_t magic_images = 0, num_images = 0, rows = 0, cols = 0;
    images.read(reinterpret_cast<char *>(&magic_images), 4);
    images.read(reinterpret_cast<char *>(&num_images), 4);
    images.read(reinterpret_cast<char *>(&rows), 4);
    images.read(reinterpret_cast<char *>(&cols), 4);

    // Leer cabecera de etiquetas
    int32_t magic_labels = 0, num_labels = 0;
    labels.read(reinterpret_cast<char *>(&magic_labels), 4);
    labels.read(reinterpret_cast<char *>(&num_labels), 4);

    // Convertir endian
    magic_images = __builtin_bswap32(magic_images);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);
    magic_labels = __builtin_bswap32(magic_labels);
    num_labels = __builtin_bswap32(num_labels);

    assert(magic_images == 2051);
    assert(magic_labels == 2049);
    assert(num_images == num_labels);

    int image_size = rows * cols;
    unsigned char pixel;
    unsigned char label;

    if (limit < 0 || limit > num_images)
        limit = num_images;

    for (int i = 0; i < limit; ++i)
    {
        // Leer imagen
        vector<float> pixels(image_size);
        for (int j = 0; j < image_size; ++j)
        {
            images.read(reinterpret_cast<char *>(&pixel), 1);
            pixels[j] = pixel / 255.0f; // Normalizar a [0,1]
        }

        // Leer etiqueta
        labels.read(reinterpret_cast<char *>(&label), 1);

        // Escribir pixeles
        for (int j = 0; j < image_size; ++j)
        {
            output << fixed << setprecision(6) << pixels[j];
            output << (j == image_size - 1 ? "," : ",");
        }

        // Escribir salida one-hot
        for (int j = 0; j < 10; ++j)
        {
            output << (j == label ? "1" : "0");
            if (j < 9)
                output << ",";
        }

        output << "\n";
    }

    cout << "Archivo CSV generado con " << limit << " ejemplos: " << output_csv << endl;
}

int main()
{
    convert_mnist_to_csv(
        "database/train-images.idx3-ubyte",
        "database/train-labels.idx1-ubyte",
        "database/mnist_train_flat_3.csv",
        60000 // o un numero menor para pruebas
    );

    convert_mnist_to_csv(
        "database/t10k-images.idx3-ubyte", // imagenes de prueba
        "database/t10k-labels.idx1-ubyte", // etiquetas de prueba
        "database/mnist_test_flat.csv",    // archivo de salida
        10000                              // o un numero menor para pruebas
    );

    return 0;
}
