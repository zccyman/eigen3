import numpy as np
import onnxruntime as rt

available_providers_without_tvm_and_tensorrt = [
    provider for provider in rt.get_available_providers()
    if provider not in {"TvmExecutionProvider", "TensorrtExecutionProvider"}
]
custom_op_model = "test/testdata/custom_op_test.onnx"
shared_library = "libs/libcustom_op_library_cpu.so"
# shared_library = "/workspace/ssy/onnxruntime/build/Linux/Release/libcustom_op_library.so"
so1 = rt.SessionOptions()
so1.register_custom_ops_library(shared_library)

# Model loading successfully indicates that the custom op node could be resolved successfully
sess1 = rt.InferenceSession(
    custom_op_model,
    sess_options=so1,
    providers=available_providers_without_tvm_and_tensorrt)
# Run with input data
input_name_0 = sess1.get_inputs()[0].name
input_name_1 = sess1.get_inputs()[1].name
output_name = sess1.get_outputs()[0].name
input_0 = np.ones((3, 5)).astype(np.float32)
input_1 = np.zeros((3, 5)).astype(np.float32)
res = sess1.run([output_name], {input_name_0: input_0, input_name_1: input_1})
output_expected = np.ones((3, 5)).astype(np.float32)
np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

# Create an alias of SessionOptions instance
# We will use this alias to construct another InferenceSession
so2 = so1

# Model loading successfully indicates that the custom op node could be resolved successfully
sess2 = rt.InferenceSession(
    custom_op_model,
    sess_options=so2,
    providers=available_providers_without_tvm_and_tensorrt)

# Create another SessionOptions instance with the same shared library referenced
so3 = rt.SessionOptions()
so3.register_custom_ops_library(shared_library)
sess3 = rt.InferenceSession(
    custom_op_model,
    sess_options=so3,
    providers=available_providers_without_tvm_and_tensorrt)
