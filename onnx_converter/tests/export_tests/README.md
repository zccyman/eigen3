# Unittest

## export_tests

### package and install onnx_converter
```
<!-- python setup.py bdist_wheel -->
<!-- cp -rf dist/onnx_converter-2.0.3-cp37-cp37m-linux_x86_64.whl export_tests -->
pip install --force-reinstall onnx_converter-2.0.3-cp37-cp37m-linux_x86_64.whl
```

### test_single_layer_export
```
pytest export_tests/test_single_layer_export.py
```

### test_all_layer_export
```
pytest export_tests/test_all_layer_export.py
```

### Quantization methods supported by converter 2.0

| Layertype     | sym_pertensor | sym_perchannel | asym_pertensor | asym_perchannel |
| ------------- | ------------- | -------------- | -------------- | --------------- |
| conv          | Y             | Y              | Y              | Y               |
| depthwiseconv | Y             | Y              | Y              | Y               |
| fc            | Y             | Y              | Y              | Y               |
| lstm          | Y             | N              | N              | N               |

| Unittest | Layertype         | intscale | floatscale | shiftfloatscale | ffloatscale | float | smooth | table | preintscale |
| -------- | ----------------- | -------- | ---------- | --------------- | ----------- | ----- | ------ | ----- | ----------- |
| Y        | conv              | Y        | Y          | Y               | N           | N     | N      | N     | N           |
| Y        | depthwiseconv     | Y        | Y          | Y               | N           | N     | N      | N     | N           |
| Y        | fc                | Y        | Y          | Y               | N           | N     | N      | N     | N           |
| Y        | pmul              | Y        | Y          | N               | Y           | Y     | N      | N     | N           |
| Y        | cmul              | Y        | Y          | N               | Y           | Y     | N      | N     | N           |
| Y        | lstm              | N        | N          | N               | Y           | N     | N      | N     | N           |
| Y        | layernormlization | N        | N          | N               | N           | Y     | N      | N     | N           |
| Y        | batchnormlization | N        | N          | N               | N           | Y     | N      | N     | N           |
| Y        | resize            | N        | N          | N               | N           | Y     | N      | N     | N           |
| Y        | averagepool       | N        | N          | N               | N           | N     | Y      | N     | N           |
| Y        | maxpool           | N        | N          | N               | N           | N     | Y      | N     | N           |
| Y        | shuffle_only      | N        | N          | N               | N           | N     | Y      | N     | N           |
| Y        | shuffle           | N        | N          | N               | N           | N     | N      | N     | Y           |
| Y        | add               | N        | N          | N               | N           | N     | N      | N     | Y           |
| Y        | sub               | N        | N          | N               | N           | N     | N      | N     | Y           |
| N        | cadd              | N        | N          | N               | N           | N     | N      | N     | Y           |
| N        | csub              | N        | N          | N               | N           | N     | N      | N     | Y           |
| Y        | concat            | N        | N          | N               | N           | N     | N      | N     | Y           |
| Y        | split             | N        | N          | N               | N           | N     | N      | N     | Y           |
| N        | relu              | N        | N          | N               | N           | N     | N      | N     | Y           |
| N        | relu6             | N        | N          | N               | N           | N     | N      | N     | Y           |
| N        | relux             | N        | N          | N               | N           | N     | N      | N     | Y           |
| Y        | leakyrelu         | N        | N          | N               | N           | N     | N      | Y     | N           |
| Y        | sigmoid           | N        | N          | N               | N           | Y     | N      | Y     | N           |
| Y        | tanh              | N        | N          | N               | N           | Y     | N      | Y     | N           |
| Y        | hardsigmoid       | N        | N          | N               | N           | Y     | N      | Y     | N           |
| Y        | hardswish         | N        | N          | N               | N           | Y     | N      | Y     | N           |
| N        | hardtanh          | N        | N          | N               | N           | Y     | N      | Y     | N           |
| N        | hardshrink        | N        | N          | N               | N           | Y     | N      | Y     | N           |