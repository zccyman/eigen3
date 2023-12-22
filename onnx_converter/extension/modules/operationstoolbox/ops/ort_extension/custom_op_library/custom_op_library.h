
#include "onnxruntime_cxx_api.h"

#include <vector>
#include <cmath>
#include <mutex>

struct OrtTensorDimensions : std::vector<int64_t>
{
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value)
    {
        OrtTensorTypeAndShapeInfo *info = ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};

struct KernelOne
{
    KernelOne(const OrtApi &api)
        : ort_(api)
    {
    }

    void Compute(OrtKernelContext *context)
    {
        // Setup inputs
        const OrtValue *input_X = ort_.KernelContext_GetInput(context, 0);
        const OrtValue *input_Y = ort_.KernelContext_GetInput(context, 1);
        const float *X = ort_.GetTensorData<float>(input_X);
        const float *Y = ort_.GetTensorData<float>(input_Y);

        // Setup output
        OrtTensorDimensions dimensions(ort_, input_X);

        OrtValue *output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
        float *out = ort_.GetTensorMutableData<float>(output);

        OrtTensorTypeAndShapeInfo *output_info = ort_.GetTensorTypeAndShape(output);
        int64_t size = ort_.GetTensorShapeElementCount(output_info);
        ort_.ReleaseTensorTypeAndShapeInfo(output_info);

        // Do computation
#ifdef WITH_CUDA
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(ort_.KernelContext_GetGPUComputeStream(context));
        cuda_add(size, out, X, Y, stream);
#else
        for (int64_t i = 0; i < size; i++)
        {
            out[i] = X[i] + Y[i];
        }
#endif
    }

private:
    Ort::CustomOpApi ort_;
};

struct KernelTwo
{
    KernelTwo(const OrtApi &api)
        : ort_(api)
    {
    }

    void Compute(OrtKernelContext *context)
    {
        // Setup inputs
        const OrtValue *input_X = ort_.KernelContext_GetInput(context, 0);
        const float *X = ort_.GetTensorData<float>(input_X);

        // Setup output
        OrtTensorDimensions dimensions(ort_, input_X);

        OrtValue *output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
        int32_t *out = ort_.GetTensorMutableData<int32_t>(output);

        OrtTensorTypeAndShapeInfo *output_info = ort_.GetTensorTypeAndShape(output);
        int64_t size = ort_.GetTensorShapeElementCount(output_info);
        ort_.ReleaseTensorTypeAndShapeInfo(output_info);

        // Do computation
        for (int64_t i = 0; i < size; i++)
        {
            out[i] = (int32_t)(round(X[i]));
        }
    }

private:
    Ort::CustomOpApi ort_;
};

struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne>
{
    void *CreateKernel(const OrtApi &api, const OrtKernelInfo * /* info */) const
    {
        return new KernelOne(api);
    };

    const char *GetName() const { return "CustomOpOne"; };

#ifdef WITH_CUDA
    const char *GetExecutionProviderType() const
    {
        return "CUDAExecutionProvider";
    };
#endif

    size_t GetInputTypeCount() const
    {
        return 2;
    };
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

    size_t GetOutputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

};

struct CustomOpTwo : Ort::CustomOpBase<CustomOpTwo, KernelTwo>
{
    void *CreateKernel(const OrtApi &api, const OrtKernelInfo * /* info */) const
    {
        return new KernelTwo(api);
    };

    const char *GetName() const { return "CustomOpTwo"; };

    size_t GetInputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

    size_t GetOutputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };

};