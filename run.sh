#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

case "$1" in
  train)
    echo "Compiling and running 'train' mode..."
    g++ main.cpp -o main -Iinclude -I. -lm -fopenmp -O3 && ./main
    ;;
  view)
    echo "Compiling and running 'view' mode (visualizer)..."
    g++ visualizer.cpp -o visualizer -Iinclude -I. -lglut -lGL -lGLU -lm -fopenmp && ./visualizer
    ;;
  cmake)
    echo "Starting CMake build process..."
    if [ -d build ]; then
        echo "Cleaning previous build directory..."
        rm -rf build
    fi

    mkdir -p build # Use -p to avoid error if build already exists (though rm -rf handles this)
    cd build
    cmake ..
    make -j$(nproc) # Use all available CPU cores for compilation
    echo "Build completed successfully."

    cd .. # Go back to the parent directory
    echo "Running the 'kohonen' executable from the build directory..."
    ./build/kohonen # Execute the kohonen program built by CMake
    ;;
  *)
    echo "Usage: $0 {train|view|cmake}"
    echo "  train: Compiles and runs the main training program."
    echo "  view: Compiles and runs the visualizer program."
    echo "  cmake: Performs a CMake build, compiles the project, and then runs 'build/kohonen'."
    exit 1
    ;;
esac