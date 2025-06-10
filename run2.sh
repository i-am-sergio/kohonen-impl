#!/bin/bash

case "$1" in
  train)
    g++ main.cpp -o main -Isrc -I. -lglut -lGL -lGLU -lm -fopenmp -O3 && ./main
    ;;
  view)
    g++ visualizer.cpp -o visualizer -Iinclude -I. -lglut -lGL -lGLU -lm -fopenmp && ./visualizer
    ;;
  *)
    echo "Usage: $0 {kohonen|visualizer}"
    exit 1
    ;;
esac