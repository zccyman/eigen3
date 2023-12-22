#!/bin/bash

cd operationstoolbox
# bash configure.sh [PYTHON_EXECUTABLE] [pybind11_DIR] [USE_CUDA] [CUDA_VERSION]
# bash configure.sh `which python` `pybind11-config --cmakedir` OFF 11.2
bash configure.sh $1 $2 $3 $4
cd ../quantizetoolbox
bash configure.sh $1 $2 $3 $4
