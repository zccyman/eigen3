#include <stdio.h>
#include <atomic>
#include <stdexcept>
#include <iostream>
#include <string>
#include "gtest/gtest.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"
#include "onnxruntime_run_options_config_keys.h"

#ifdef _WIN32
#define ORT_UNUSED_PARAMETER(x) (x)
#else
#define ORT_UNUSED_PARAMETER(x) (void)(x)
#endif

struct MockedOrtAllocator : OrtAllocator {
    MockedOrtAllocator();
    ~MockedOrtAllocator();

    void *Alloc(size_t size);
    void Free(void *p);
    const OrtMemoryInfo *Info() const;
    size_t NumAllocations() const;

    void LeakCheck();

private:
    MockedOrtAllocator(const MockedOrtAllocator &) = delete;
    MockedOrtAllocator &operator=(const MockedOrtAllocator &) = delete;

    std::atomic<size_t> memory_inuse{0};
    std::atomic<size_t> num_allocations{0};
    OrtMemoryInfo *cpu_memory_info;
};

MockedOrtAllocator::MockedOrtAllocator()
{
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator *this_, size_t size) {
        return static_cast<MockedOrtAllocator *>(this_)->Alloc(size);
    };
    OrtAllocator::Free = [](OrtAllocator *this_, void *p) {
        static_cast<MockedOrtAllocator *>(this_)->Free(p);
    };
    OrtAllocator::Info = [](const OrtAllocator *this_) {
        return static_cast<const MockedOrtAllocator *>(this_)->Info();
    };
    Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(
        OrtDeviceAllocator, OrtMemTypeDefault, &cpu_memory_info));
}

MockedOrtAllocator::~MockedOrtAllocator()
{
    Ort::GetApi().ReleaseMemoryInfo(cpu_memory_info);
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26400)
#pragma warning(disable : 26409)
#endif
void *MockedOrtAllocator::Alloc(size_t size)
{
    constexpr size_t extra_len = sizeof(size_t);
    memory_inuse.fetch_add(size += extra_len);
    void *p = new (std::nothrow) uint8_t[size];
    if (p == nullptr)
        return p;
    num_allocations.fetch_add(1);
    *(size_t *)p = size;
    return (char *)p + extra_len;
}

void MockedOrtAllocator::Free(void *p)
{
    constexpr size_t extra_len = sizeof(size_t);
    if (!p)
        return;
    p = (char *)p - extra_len;
    size_t len = *(size_t *)p;
    memory_inuse.fetch_sub(len);
    delete[] reinterpret_cast<uint8_t *>(p);
}

const OrtMemoryInfo *MockedOrtAllocator::Info() const
{
    return cpu_memory_info;
}

size_t MockedOrtAllocator::NumAllocations() const
{
    return num_allocations.load();
}

void MockedOrtAllocator::LeakCheck()
{
    if (memory_inuse.load()) {
        std::cerr << "memory leak!!!" << std::endl;
        exit(-1);
    }
}

struct Input {
    const char *name = nullptr;
    std::vector<int64_t> dims;
    std::vector<float> values;
};

template <typename OutT>
void RunSession(OrtAllocator *allocator,
    Ort::Session &session_object,
    const std::vector<Input> &inputs,
    const char *output_name,
    const std::vector<int64_t> &dims_y,
    const std::vector<OutT> &values_y,
    Ort::Value *output_tensor)
{
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char *> input_names;
    for (size_t i = 0; i < inputs.size(); i++) {
        input_names.emplace_back(inputs[i].name);
        ort_inputs.emplace_back(
            Ort::Value::CreateTensor<float>(allocator->Info(allocator),
                const_cast<float *>(inputs[i].values.data()),
                inputs[i].values.size(), inputs[i].dims.data(),
                inputs[i].dims.size()));
    }

    std::vector<Ort::Value> ort_outputs;
    if (output_tensor)
        session_object.Run(Ort::RunOptions{nullptr}, input_names.data(),
            ort_inputs.data(), ort_inputs.size(), &output_name, output_tensor,
            1);
    else {
        ort_outputs = session_object.Run(Ort::RunOptions{}, input_names.data(),
            ort_inputs.data(), ort_inputs.size(), &output_name, 1);
        ASSERT_EQ(ort_outputs.size(), 1u);
        output_tensor = &ort_outputs[0];
    }

    auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), dims_y);
    size_t total_len = type_info.GetElementCount();
    ASSERT_EQ(values_y.size(), total_len);

    OutT *f = output_tensor->GetTensorMutableData<OutT>();
    for (size_t i = 0; i != total_len; ++i) {
        ASSERT_EQ(values_y[i], f[i]);
    }
}

template <typename OutT>
static void TestInference(Ort::Env &env,
    const std::string &model_uri,
    const std::vector<Input> &inputs,
    const char *output_name,
    const std::vector<int64_t> &expected_dims_y,
    const std::vector<OutT> &expected_values_y,
    int provider_type,
    OrtCustomOpDomain *custom_op_domain_ptr,
    const char *custom_op_library_filename,
    void **library_handle = nullptr,
    bool test_session_creation_only = false,
    void *cuda_compute_stream = nullptr)
{
    Ort::SessionOptions session_options;

    if (provider_type == 1) {
#ifdef WITH_CUDA
        std::cout << "Running simple inference with cuda provider" << std::endl;
        auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(
            cuda_compute_stream);
        session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
        ORT_UNUSED_PARAMETER(cuda_compute_stream);
        return;
#endif
    } else if (provider_type == 2) {
#ifdef USE_DNNL
        Ort::ThrowOnError(
            OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, 1));
        std::cout << "Running simple inference with dnnl provider" << std::endl;
#else
        return;
#endif
    } else if (provider_type == 3) {
#ifdef USE_NUPHAR
        Ort::ThrowOnError(
            OrtSessionOptionsAppendExecutionProvider_Nuphar(session_options,
                /*allow_unaligned_buffers*/ 1, ""));
        std::cout << "Running simple inference with nuphar provider"
                  << std::endl;
#else
        return;
#endif
    } else {
        std::cout << "Running simple inference with default provider"
                  << std::endl;
    }

    if (custom_op_library_filename) {
        Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(
            session_options, custom_op_library_filename, library_handle));
    }

    // if session creation passes, model loads fine
    Ort::Session session(env, model_uri.c_str(), session_options);

    // caller wants to test running the model (not just loading the model)
    if (!test_session_creation_only) {
        // Now run
        auto default_allocator = std::make_unique<MockedOrtAllocator>();

        // without preallocated output tensor
        RunSession<OutT>(default_allocator.get(), session, inputs, output_name,
            expected_dims_y, expected_values_y, nullptr);
        // with preallocated output tensor
        Ort::Value value_y =
            Ort::Value::CreateTensor<float>(default_allocator.get(),
                expected_dims_y.data(), expected_dims_y.size());

        // test it twice
        for (int i = 0; i != 2; ++i)
            RunSession<OutT>(default_allocator.get(), session, inputs,
                output_name, expected_dims_y, expected_values_y, &value_y);
        float *result =
            static_cast<float *>(value_y.GetTensorMutableData<float>());
        std::cout << result[0] << std::endl;
    }
}

int main(int argc, char **argv)
{
    // std::unique_ptr<Ort::Env> ort_env;
    std::string model_name = "test/testdata/custom_op_test.onnx";
    std::string lib_name = "libs/libcustom_op_library_cpu.so";
    Ort::Env env =
        Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, model_name.c_str());
    std::vector<Input> inputs(2);
    inputs[0].name = "input_1";
    inputs[0].dims = {3, 5};
    inputs[0].values = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f,
        10.0f, 11.1f, 12.2f, 13.3f, 14.4f, 15.5f};
    inputs[1].name = "input_2";
    inputs[1].dims = {3, 5};
    inputs[1].values = {15.5f, 14.4f, 13.3f, 12.2f, 11.1f, 10.0f, 9.9f, 8.8f,
        7.7f, 6.6f, 5.5f, 4.4f, 3.3f, 2.2f, 1.1f};
    std::vector<int64_t> expected_dims_y = {3, 5};
    std::vector<int32_t> expected_values_y = {
        17, 17, 17, 17, 17, 17, 18, 18, 18, 17, 17, 17, 17, 17, 17};
    void *library_handle = nullptr;
    TestInference<int32_t>(env, model_name, inputs, "output", expected_dims_y,
        expected_values_y, 0, nullptr, lib_name.c_str(), &library_handle);
    return 0;
}