#!/bin/bash

case "$1" in
  train)
    g++ main.cpp -o main -Iinclude -I. -lglut -lGL -lGLU -lm -fopenmp && ./main
    ;;
  view)
    g++ visualizer.cpp -o visualizer -Iinclude -I. -lglut -lGL -lGLU -lm -fopenmp && ./visualizer
    ;;
  *)
    echo "Usage: $0 {kohonen|visualizer}"
    exit 1
    ;;
esac