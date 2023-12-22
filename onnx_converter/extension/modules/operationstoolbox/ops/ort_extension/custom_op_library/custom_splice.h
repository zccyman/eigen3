#ifndef ONNXRUNTIME_TIMEINTELLI_SPLICE_H
#define ONNXRUNTIME_TIMEINTELLI_SPLICE_H

#include <onnxruntime_cxx_api.h>

struct TIMESINTELLISpliceKernel {
    TIMESINTELLISpliceKernel(OrtApi api, const OrtKernelInfo *info);

    void Compute(OrtKernelContext *context);

protected:
    OrtApi api_;
    Ort::CustomOpApi ort_;
    const OrtKernelInfo *info_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<int64_t> context_;
    std::vector<int64_t> forward_indexes;
    int64_t has_fc;
    std::vector<float> weight;
    std::vector<float> bias;
    int64_t output_dim;
};

struct TIMESINTELLISpliceOp
    : Ort::CustomOpBase<TIMESINTELLISpliceOp, TIMESINTELLISpliceKernel> {
    void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const
    {
        return new TIMESINTELLISpliceKernel(api, info);
    }

    const char *GetName() const
    {
        return "Splice";
    };

    size_t GetInputTypeCount() const
    {
        return 1;
    };

    ONNXTensorElementDataType GetInputType(size_t /*index*/) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };

    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
        size_t index) const
    {
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }

    size_t GetOutputTypeCount() const
    {
        return 1;
    };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };

    // force cpu
    const char *GetExecutionProviderType() const
    {
        return "CPUExecutionProvider";
    };
};
#endif