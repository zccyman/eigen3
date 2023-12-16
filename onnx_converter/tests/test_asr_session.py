import os
import copy
import onnx
import numpy as np
import onnxruntime as rt

import torch
import torchaudio
import soundfile
import librosa

# from ort_inference import ORTSession, ORTTestInference


def LoadCustomOps(
        custom_op_model="/buffer/ssy/splice/ort_extension/test_model_dir/decode_v11_adj_bn.onnx",
        shared_library="./build/custom_op_library/libcustom_op_library_cpu.so"):
    rt.set_default_logger_severity(3)

    custom_op_model = "/buffer/ssy/splice/ort_extension/decode_v11_adj_bn.onnx"#mdl_0621_without_lda_adj_domain, decode_v11_adj
    model = onnx.load(custom_op_model)
    node_outputs = []
    for node in model.graph.node:
        node_outputs.extend(node.output)
        if node.op_type in ["Splice"]:
            node.domain = "timesintelli.com"
    names = [node.name for node in model.graph.node]

    onnx.save(model, custom_op_model.replace(".onnx", "_domain.onnx"))

    # custom_op_model = custom_op_model.replace(".onnx", "_domain.onnx")

    # model = onnx.load(custom_op_model)
    for output in node_outputs:
        model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    # shared_library = ["extension/modules/operationstoolbox/libs/libcustom_op_library_cpu.so"]
    shared_library = ["extension/libs/libcustom_op_library_cpu.so"]
    # sess_option = ORTSession(model=custom_op_model, shared_library=shared_library, providers=["CUDAExecutionProvider"])
    # sess_option = ORTSession(model=custom_op_model, shared_library=shared_library, providers=["CPUExecutionProvider"])
    # sess_option = ORTSession(
    #     model=model.SerializeToString(),
    #     shared_library=shared_library,
    #     providers=["CPUExecutionProvider"])

    def get_sess_options(shared_librarys):
        import onnxruntime as rt
        sess_options = rt.SessionOptions()
        for shared_library in shared_librarys:
            sess_options.register_custom_ops_library(shared_library)
        return sess_options

    # Model loading successfully indicates that the custom op node could be resolved successfully
    sess1 = rt.InferenceSession(
        model.SerializeToString(), sess_options=get_sess_options(shared_library), providers=["CPUExecutionProvider"]
    )

    return sess1, node_outputs
# Run with input data

def read_txt(file_name="feats_1.txt", context=','):
    data = []
    with open(file_name) as in_f:
        lines = in_f.readlines()

        for line in lines:
            for num in line.split(context):
                if num == '\n' or num == "":
                    continue
                data.append(np.float32(num))
    return data

def wav_read(filename, tgt_fs=None):
    y, fs = soundfile.read(filename, dtype='float32')
    if tgt_fs is not None:
        if fs != tgt_fs:
            if fs != 16000:
                y = librosa.resample(y, tgt_fs, 16000)
                fs = tgt_fs
    return y, fs


def VoiceMFCC(wav_file, sample_frequency=16000, num_ceps=40, low_freq=20, high_freq=-400):
    in_data = torchaudio.load(wav_file)
    mfcc_data = torchaudio.compliance.kaldi.mfcc(
            in_data[0],
            sample_frequency=sample_frequency,
            num_ceps=num_ceps,
            num_mel_bins=num_ceps,
            low_freq=low_freq,
            high_freq=high_freq).numpy()
    kaldi_feat = np.array(read_txt(os.path.join("/buffer/ssy/splice/ort_extension/test_model_dir", "feats_1.txt")), dtype=np.float32).reshape(-1, 43)
    diff = kaldi_feat - mfcc_data
    return mfcc_data

def VoicefBank():
    pass

def CompareDiff(fd_path, start_c, ref_path, session_output, output_size, left_chunk_size, right_chunk_size, context=" "):
    onnx_output = session_output[left_chunk_size:]
    ref_output = np.array(read_txt(os.path.join(fd_path, ref_path), context), dtype=np.float32).reshape(-1, output_size)
    ref_output = ref_output[start_c:start_c+onnx_output.shape[0]]
    diff = ref_output - onnx_output[:ref_output.shape[0]]
    pb = diff/onnx_output[:ref_output.shape[0]]
    pb[np.isnan(pb)] = 0
    return diff, pb

def ASRORTInference(data, chunk_size, feat_size, in_size, left_chunk_size, right_chunk_size, output, fd_path="./"):
    sess1, node_outputs = LoadCustomOps()
    in_data = np.array(data, dtype=np.float32).reshape(-1, 43)
    n_block = (chunk_size-left_chunk_size)
    frame_size = in_data.shape[0] // n_block + 1

    session_output = []
    for idx in range(frame_size):
        # start_c, end_c = chunk_size*idx, chunk_size*idx + chunk_size
        start_c, end_c = n_block*idx, n_block*idx + chunk_size
        end_c = np.clip(end_c, 0, in_data.shape[0])
        data = in_data[start_c:end_c]
        input_data = np.zeros((in_size, feat_size), dtype=np.float32)
        start = start_c - left_chunk_size
        end = end_c + right_chunk_size
        if start < 0 and end_c < in_data.shape[0]-1:
            input_data[:-start] = in_data[0]
            input_data[-start:] = in_data[0:end]
        elif end_c <= in_data.shape[0]-1:
            input_data = in_data[start:end]
        else:
            input_data[:in_data.shape[0]-start] = in_data[start:]
            input_data[in_data.shape[0]-start:] = in_data[in_data.shape[0]-1]
        input_name_0 = sess1.get_inputs()[0].name
        output_name = sess1.get_outputs()[0].name
        res = sess1.run(node_outputs, {input_name_0: input_data})
        if True:#idx == 0:
            b, pb = CompareDiff(fd_path, start_c, "ouput_tdnn1.affine.txt", res[1], 80, left_chunk_size, right_chunk_size)
            c, pc = CompareDiff(fd_path, start_c, "ouput_tdnn1.relu.txt", res[2], 80, left_chunk_size, right_chunk_size)
            d, pd = CompareDiff(fd_path, start_c, "ouput_tdnn1.batchnorm.txt", res[3], 80, left_chunk_size, right_chunk_size)
            e, pe = CompareDiff(fd_path, start_c, "ouput_tdnn2.affine.txt", res[5], 80, left_chunk_size, right_chunk_size)
            f, pf = CompareDiff(fd_path, start_c, "ouput_tdnn3.affine.txt", res[9], 80, left_chunk_size, right_chunk_size)
            g, pg = CompareDiff(fd_path, start_c, "ouput_tdnn4.affine.txt", res[13], 80, left_chunk_size, right_chunk_size)
            h, ph = CompareDiff(fd_path, start_c, "ouput_tdnn4.relu.txt", res[14], 80, left_chunk_size, right_chunk_size)
            i, pi = CompareDiff(fd_path, start_c, "ouput_tdnn4.batchnorm.txt", res[15], 80, left_chunk_size, right_chunk_size)
            j, pj = CompareDiff(fd_path, start_c, "ouput_tdnn5.affine.txt", res[17], 80, left_chunk_size, right_chunk_size)
            k, pk = CompareDiff(fd_path, start_c, "ouput_tdnn6.affine.txt", res[21], 80, left_chunk_size, right_chunk_size)
            l, pl = CompareDiff(fd_path, start_c, "ouput_tdnn6.batchnorm.txt", res[23], 80, left_chunk_size, right_chunk_size)
            m, pm = CompareDiff(fd_path, start_c, "output_prefinal-chain.affine.txt", res[24], 80, left_chunk_size, right_chunk_size)
            n, pn = CompareDiff(fd_path, start_c, "output_prefinal-chain.relu.txt", res[25], 80, left_chunk_size, right_chunk_size)
            o, po = CompareDiff(fd_path, start_c, "output_prefinal-chain.batchnorm.txt", res[26], 80, left_chunk_size, right_chunk_size)
            p, pp = CompareDiff(fd_path, start_c, "output.txt", res[27], 2120, left_chunk_size, right_chunk_size, context=",")
        session_output.append(res[-1][right_chunk_size:])
        # print(res[-1][left_chunk_size:].shape)
    results = np.array(session_output).reshape(-1, res[-1].shape[1])
    return results

def ASRInference(
    fd_path, feat_file, output_file, chunk_size=50,
    feat_size=43, in_size=74, output_size=2120,
    left_chunk_size=12, right_chunk_size=12):
    # fd_path = "./test_model_dir_without_lda"
    # fd_path = "./test_model_dir"
    data = read_txt(os.path.join(fd_path, feat_file))
    ref_out = np.array(read_txt(os.path.join(fd_path, output_file))).reshape(-1, output_size)

    results = ASRORTInference(data, chunk_size, feat_size, in_size,
                            left_chunk_size, right_chunk_size,
                            ref_out, fd_path=fd_path)
    # print()
    diff = results[:ref_out.shape[0]] - ref_out
    print(diff.min(), diff.max())
    print("Done!")

if __name__=="__main__":
    # fd_path = "./test_model_dir_without_lda"
    fd_path = "/buffer/ssy/splice/ort_extension/test_model_dir"
    feat_file = "feats_1.txt"
    output_file = "output.txt"
    # lda_feat = np.array(read_txt(os.path.join(fd_path, "ouput_lda.txt")), dtype=np.float32).reshape(-1, 129)
    # mat = np.array(read_txt(os.path.join(fd_path, "lda.mat.txt")), dtype=np.float32).reshape(129, -1)
    chunk_size = 50
    feat_size = 43
    in_size = 74
    output_size = 2120
    left_chunk_size, right_chunk_size = 12, 12
    ASRInference(fd_path, feat_file, output_file, chunk_size, feat_size, in_size, output_size, left_chunk_size, right_chunk_size)
    # VoiceMFCC("/buffer/ssy/splice/ort_extension/chengqian_26.wav")
