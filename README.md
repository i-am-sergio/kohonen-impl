# Red Neuronal de Kohonen

## 1. Test 1: Imprimir imagen por consola (binarizacion)

```sh
g++ test.cpp -o a && ./a
```

## 2. Test 2: Visualizar imagen con OpenGL + Animacion

```sh
g++ test2.cpp -o b -I. -I/usr/include -L/usr/lib -lglut -lGL -lGLU && ./b
```

## Generate CSV archives of MNIST

```bash
make convert
```

## Execute Kohonen with MNIST

```bash
make run
./build/kohonen
```

## Execute View Sphere Neuron Kohonen with MNIST image

```bash
./run3.sh
```
