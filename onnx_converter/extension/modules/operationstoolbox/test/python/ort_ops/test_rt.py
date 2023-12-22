import onnx
import onnxruntime as rt

option = rt.SessionOptions()
option.register_custom_ops_library('libs/libcustom_op_library_cpu.so')
model = onnx.load('test/testdata/custom_op_test.onnx')
sess = rt.InferenceSession(model.SerializeToString(), option)