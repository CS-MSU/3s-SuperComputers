#!/bin/bash

module load SpectrumMPI/10.1.0

MPICXX=mpixlC

echo "Compile"
${MPICXX} -fopenmp src/main.cpp src/Matrix.cpp src/PuassonEquation.cpp -o app -std=c++11
