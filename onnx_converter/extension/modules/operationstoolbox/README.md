# operationstoolbox

#### Compilation
```
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2

# compile googletest
# git clone https://github.com/google/googletest.git
wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz
tar -zxvf release-1.8.0.tar.gz
cd googletest-release-1.8.0 && rm -rf ./build && mkdir build && cd build && cmake .. && make -j4 && make install
cd ../../

# nsync
git clone https://github.com/google/nsync.git

bash configure.sh [PYTHON_EXECUTABLE] [pybind11_DIR] [USE_CUDA] [CUDA_VERSION]  or bash configure.sh [PYTHON_EXECUTABLE] [pybind11_DIR] [USE_CUDA]
bash configure.sh `which python` `pybind11-config --cmakedir` ON 11.2 
or 
bash configure.sh `which python` `pybind11-config --cmakedir` OFF 11.2
```

### Test
pytest test/python/ops
