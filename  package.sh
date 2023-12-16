#! /bin/bash

set -e

version=$1
branch=$2

# rm -rf onnx-converter
# git clone https://192.168.1.240/vision-algorithm/onnx-converter.git -b ${branch}
git config -f .gitmodules submodule.onnx-converter.branch ${branch}
git submodule update --init --recursive --force
git submodule update --remote

source activate base
conda install python=3.7 -y
conda remove -n package --all -y
conda create -n package python=3.7 -y
source activate package

# install eigen3-dev
apt install libeigen3-dev

# compile googletest
# git clone https://github.com/google/googletest.git
wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz
tar -zxvf release-1.8.0.tar.gz
cd googletest-release-1.8.0 && rm -rf ./build && mkdir build && cd build && cmake .. && make -j4 && make install
cd ../../

pip install -r customer_release/requirement.txt
cd onnx-converter
pip install -r benchmark/requirement.txt
python simulator/perf_analysis.py
cd extension/modules
bash configure.sh `which python` `pybind11-config --cmakedir` OFF 11.2
cd ../../../
python setup.py bdist_wheel
# cat /root/miniconda3/envs/package/lib/python3.7/site-packages/onnx_converter/utest/data/commit_id.txt 

rm -rf onnx-converter/checkpoint 
rm -rf onnx-converter/export 
rm -rf onnx-converter/graph 
rm -rf onnx-converter/quantizer
rm -rf onnx-converter/simulator
rm -rf onnx-converter/utest
rm -rf onnx-converter/tools
rm -rf onnx-converter/utils
rm -rf onnx-converter/OnnxConverter.py
rm -rf onnx-converter/__init__.py

pip install --force-reinstall dist/onnx_converter-${version}-cp37-cp37m-linux_x86_64.whl
python check_version.py ${version}
