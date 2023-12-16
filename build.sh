cd onnx_converter
pip install -r benchmark/requirement.txt
python simulator/perf_analysis.py
cd extension/modules
bash configure.sh `which python` `pybind11-config --cmakedir` OFF 11.2
cd ../../../
python setup.py bdist_wheel