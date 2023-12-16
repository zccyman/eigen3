# operationstoolbox

#### Compilation
```
bash configure.sh [PYTHON_EXECUTABLE] [pybind11_DIR] [USE_CUDA] [CUDA_VERSION]  or bash configure.sh [PYTHON_EXECUTABLE] [pybind11_DIR] [USE_CUDA]
bash configure.sh `which python` `pybind11-config --cmakedir` ON 11.2 
or 
bash configure.sh `which python` `pybind11-config --cmakedir` OFF 11.2
```

### Test
pytest test/python/ops
