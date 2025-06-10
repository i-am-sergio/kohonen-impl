#!/bin/bash

case "$1" in
  kohonen)
    g++ -O3 -fopenmp kohonen.cpp -o kohonen && ./kohonen
    ;;
  viewneuron)
    g++ viewneuron.cpp -o viewneuron -Iinclude -I. -lglut -lGL -lGLU -lm -fopenmp && ./viewneuron
    ;;
  *)
    echo "Usage: $0 {kohonen|viewneuron}"
    exit 1
    ;;
esac