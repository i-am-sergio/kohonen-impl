# Red Neuronal de Kohonen

<p align="center">
    <img src="img/image.png" width="600">
</p>

Este proyecto implementa una **Red Neuronal de Kohonen** (también conocida como **Mapa Autoorganizado** o **SOM - Self-Organizing Map**) entrenada con el dataset MNIST para representar visualmente los patrones de dígitos manuscritos.
La red aprende a agrupar vectores de entrada similares en ubicaciones vecinas dentro de una cuadrícula bidimensional, permitiendo una representación topológica del conjunto de datos.

---

## ¿Cómo funciona una Red de Kohonen?

Una red de Kohonen consta de una capa de salida organizada en una malla (por ejemplo, 10x10) donde cada neurona representa un "peso" o prototipo que compite por activarse según la similitud con la entrada.
El entrenamiento se realiza de forma no supervisada a través de los siguientes pasos:

1. **Inicialización**: Los pesos de cada neurona se inicializan aleatoriamente.
2. **Competencia**: Por cada entrada, se determina la neurona ganadora (Best Matching Unit o **BMU**) cuya distancia al vector de entrada es mínima.
3. **Actualización**: La BMU y sus vecinas actualizan sus pesos para parecerse más al vector de entrada.
4. **Decaimiento**: La tasa de aprendizaje y el radio de vecindad disminuyen con el tiempo para estabilizar la red.

Las principales funciones utilizadas en la implementación incluyen:

* `calcularDistanciaEuclidiana`: calcula la distancia entre el vector de entrada y los pesos de cada neurona.
* `encontrarBMU`: identifica la neurona más cercana al vector de entrada.
* `actualizarPesos`: ajusta los pesos de la BMU y sus vecinos.
* `entrenarRed`: ejecuta múltiples épocas de entrenamiento con ejemplos del dataset.

---

## Requisitos

* g++
* make
* Dataset MNIST en formato binario (`.idx`)
* Sistema Unix-like

---

## 1. Convertir Dataset MNIST a CSV

Convierte el conjunto de datos MNIST desde su formato binario original (`.idx`) a archivos CSV legibles:

```bash
g++ ./convert.cpp -o convert && ./convert
```

Esto generará archivos `.csv` que contienen las imágenes y etiquetas para entrenamiento y prueba.

---

## 2. Ejecutar Entrenamiento de Red Kohonen

Este paso entrena la red neuronal con los datos convertidos:

```bash
./run.sh train
```

Durante el entrenamiento, los pesos de la red se ajustan gradualmente para formar agrupaciones de datos similares.

---

## 3. Ejecutar Visualización de la Red Kohonen

Una vez finalizado el entrenamiento, se puede visualizar la topología aprendida por la red:

```bash
./run.sh view
```

Esto genera una representación visual que muestra cómo la red ha agrupado los diferentes dígitos del MNIST en la cuadrícula.

