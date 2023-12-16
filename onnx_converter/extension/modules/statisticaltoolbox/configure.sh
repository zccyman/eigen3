#!/bin/bash

echo "shell parameter nums: $#"

if [ $# -lt 2 ]; then
    exit -1
elif [ $# == 2 ]; then
    python_executable=$1
    pybind11_dir=$2
    use_gpu=OFF
    cuda_version="cuda"  
elif [ $# == 3 ]; then
    python_executable=$1
    pybind11_dir=$2
    use_gpu=$3    
    cuda_version="cuda"     
elif [ $# == 4 ]; then
    python_executable=$1
    pybind11_dir=$2
    use_gpu=$3
    cuda_version="cuda-$4"
else
    echo "parameters not supported yet!!!"
    exit -1
fi

echo "CUDA version: ${cuda_version}"
echo "use_gpu: ${use_gpu}"
echo "python_executable: ${python_executable}"
echo "pybind11_dir: ${pybind11_dir}"

rm -rf bins libs build && mkdir build && cd build

if [ ${use_gpu} == ON ]; then
    echo "use gpu when complie op_extension!!!"
    cmake .. \
        -DPYTHON_INCLUDE_DIR=$(${python_executable} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(${python_executable} -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
        -DPYTHON_EXECUTABLE=${python_executable} \
        -Dpybind11_DIR=${pybind11_dir} \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/${cuda_version} \
        -DCMAKE_CUDA_COMPILER=/usr/local/${cuda_version}/bin/nvcc \
        -DCUDNN_LIBRARY_PATH=/usr/local/${cuda_version}/lib64 \
        -DCMAKE_CXX_STANDARD=17 \
        -DWITH_TEST=ON \
        -DWITH_CUDA=ON
else
    echo "only use cpu when complie op_extension!!!"
    cmake .. \
        -DPYTHON_INCLUDE_DIR=$(${python_executable} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(${python_executable} -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
        -DPYTHON_EXECUTABLE=${python_executable} \
        -Dpybind11_DIR=${pybind11_dir} \
        -DCMAKE_CXX_STANDARD=17 \
        -DWITH_TEST=ON \
        -DWITH_CUDA=OFF
fi

make -j && make install
