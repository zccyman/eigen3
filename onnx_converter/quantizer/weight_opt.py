import copy
import os

import numpy as np


def weight_correction(weight, quan_method):
    quan_method.get_quan_param(weight)
    qweight = quan_method.get_quan_data(weight)
    qweight = quan_method.get_dequan_data(qweight)

    out_c = weight.shape[0]
    mean = np.mean(weight.reshape(out_c, -1))
    qmean = np.mean(qweight.reshape(out_c, -1))
    std = (weight.reshape(out_c, -1) - mean).std()
    qstd = (qweight.reshape(out_c, -1) - qmean).std()

    mu = mean - qmean
    sigma = std / qstd

    qweight = sigma * (qweight + mu)

    return qweight

def cross_layer_equalization(layer, layers, skip_layer_names=[]):
    flag_c = False # [conv + conv + conv] + concat + [conv + conv + conv]
    flag_m = False # conv + [conv + conv + conv]
    layer_next = None
    layer_next2 = None
    layer_type = layer.get_layer_type()
    node_types = [node.get_op_type().lower() for node in layer.get_nodes()]
    if layer_type in ["concat"]:
        input_idxs, output_idxs = layer.get_input_idx(), layer.get_output_idx()
        layer_pre_list = [layers[input_idx] for input_idx in input_idxs]
        layer_next_list = [layers[output_idx] for output_idx in output_idxs]
        # layer_pre_types = [layer.get_layer_type() for layer in layer_pre_list]
        # layer_next_types = [layer.get_layer_type() for layer in layer_next_list]
        
        flag_c = True 
        for layer_pre in layer_pre_list:
            if layer_pre.get_layer_type() == "conv":
                if len(layer_pre.get_output_idx()) > 1:
                    flag_c = False
        for layer_pre in layer_pre_list:
            node_types = [node.get_op_type().lower() for node in layer_pre.get_nodes()]
            if layer_pre.get_layer_type() not in ["conv"] or 'relu' not in node_types:
                flag_c = False
        for layer_nxt in layer_next_list:                    
            if layer_nxt.get_layer_type() not in ["conv"]:
                flag_c = False
    elif layer_type in ["fc", "conv"] and 'relu' in node_types and len(layer.get_output_idx()) == 1:
        layer_next = layers[layer.get_output_idx()[0]]
        node_types = [node.get_op_type().lower()
                      for node in layer_next.get_nodes()]
        if layer_next.get_layer_type() not in ["fc", "conv", "depthwiseconv", "batchnormalization"]:
            layer_next = None
        elif len(layer_next.get_output_idx()) == 1 and layer_next.get_layer_type() == "depthwiseconv" and 'relu' in node_types:
            layer_next2 = layers[layer_next.get_output_idx()[0]]
            if layer_next2.get_layer_type() not in ["conv"]:
                layer_next2 = None
    elif layer_type in ["fc", "conv"] and 'relu' in node_types and len(layer.get_output_idx()) > 1:
        output_idxs = layer.get_output_idx()
        layer_next_list = [layers[output_idx] for output_idx in output_idxs]
        layer_next_types = [layer.get_layer_type() for layer in layer_next_list]
        flag_m = True
        for lnt in layer_next_types:
            if layer_type != lnt:
                flag_m = False
    else:
        layer_next = None
                
    # flag_c = False
    # flag_m = False
    
    if flag_c:       
        layer_pre_list.reverse()# type: ignore
        print("cross_layer_equalization concat: ", layer.get_layer_name(),
              [layer.get_layer_name() for layer in layer_pre_list], # type: ignore
              [layer.get_layer_name() for layer in layer_next_list])# type: ignore

        range_pre = []
        for layer_pre in layer_pre_list:# type: ignore
            weight0 = copy.deepcopy(layer_pre.get_layer_ops()['weights'][0])
            tmp = np.amax(np.abs(weight0), axis=(1, 2, 3))
            range_pre.append(tmp)
        range0 = np.concatenate(range_pre)
            
        weight1 = copy.deepcopy(layer_next_list[0].get_layer_ops()['weights'][0])# type: ignore
        range1 = np.amax(np.abs(weight1), axis=(0, 2, 3))
        for layer_next in layer_next_list[1:]:# type: ignore
            weight2 = copy.deepcopy(layer_next.get_layer_ops()['weights'][0])
            range2 = np.amax(np.abs(weight2), axis=(0, 2, 3))
            range1 = np.maximum(range1, range2)
        scale_factor = range0 / np.sqrt(range0 * range1)
                    
        delta = 0
        for layer_pre in layer_pre_list:# type: ignore
            weight0 = copy.deepcopy(layer_pre.get_layer_ops()['weights'][0])
            bias0 = copy.deepcopy(layer_pre.get_layer_ops()['weights'][1])            
            for i in range(bias0.shape[0]):
                weight0[i, :, :, :] /= scale_factor[i + delta]
                bias0[i] /= scale_factor[i + delta]
            layer_pre.set_layer_ops(dict(weights=[weight0, bias0]))
            delta += bias0.shape[0]
            
        for layer_next in layer_next_list:# type: ignore
            weight1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][0])
            weight1_orig = copy.deepcopy(weight1)
            has_bias = layer_next.get_layer_ops()['attrs'][0]['bias']
            if has_bias:
                bias1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][1])

            for i in range(len(scale_factor)):
                weight1[:, i, :, :] *= scale_factor[i]

            layer_next.set_layer_ops(dict(weights=[weight1, bias1]))  
    elif flag_m:
        print("cross_layer_equalization2: ", layer.get_layer_name(), [layer.get_layer_name() for layer in layer_next_list])
        weight0 = copy.deepcopy(layer.get_layer_ops()['weights'][0])
        weight0_orig = copy.deepcopy(weight0)
        has_bias = layer.get_layer_ops()['attrs'][0]['bias']
        if has_bias:
            bias0 = copy.deepcopy(layer.get_layer_ops()['weights'][1])
        
        range0 = np.amax(np.abs(weight0), axis=(1, 2, 3))
        weight1 = copy.deepcopy(layer_next_list[0].get_layer_ops()['weights'][0])# type: ignore
        range1 = np.amax(np.abs(weight1), axis=(0, 2, 3))
        for layer_next in layer_next_list[1:]:# type: ignore
            weight2 = copy.deepcopy(layer_next.get_layer_ops()['weights'][0])
            range2 = np.amax(np.abs(weight2), axis=(0, 2, 3))
            range1 = np.maximum(range1, range2)
        scale_factor = range0 / np.sqrt(range0 * range1)
                    
        only_set_onces = [False for _ in layer_next_list]# type: ignore
        only_set_onces[0] = True
        for layer_next, only_set_once in zip(layer_next_list, only_set_onces):# type: ignore
            weight1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][0])
            weight1_orig = copy.deepcopy(weight1)
            has_bias = layer_next.get_layer_ops()['attrs'][0]['bias']
            if has_bias:
                bias1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][1])

            for i in range(len(scale_factor)):
                if only_set_once: 
                    weight0[i, :, :, :] /= scale_factor[i]
                    bias0[i] /= scale_factor[i]# type: ignore
                weight1[:, i, :, :] *= scale_factor[i]

            if only_set_once: 
                layer.set_layer_ops(dict(weights=[weight0, bias0]))# type: ignore
            layer_next.set_layer_ops(dict(weights=[weight1, bias1]))   # type: ignore     
    elif layer_next and layer_next2:
        flag = layer.get_layer_type() == "conv" and \
            layer_next.get_layer_type() == "depthwiseconv" and \
            layer_next2.get_layer_type() == "conv"
        if not flag:
            return
        for layer_name_ in [layer.get_layer_name(),
              layer_next.get_layer_name(), layer_next2.get_layer_name()]:
            if layer_name_ in skip_layer_names:
                return
        print("cross_layer_equalization3: ", layer.get_layer_name(),
              layer_next.get_layer_name(), layer_next2.get_layer_name())
        weight0 = copy.deepcopy(layer.get_layer_ops()['weights'][0])
        weight0_orig = copy.deepcopy(weight0)
        has_bias = layer.get_layer_ops()['attrs'][0]['bias']
        if has_bias:
            bias0 = copy.deepcopy(layer.get_layer_ops()['weights'][1])

        weight1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][0])
        weight1_orig = copy.deepcopy(weight1)
        has_bias = layer_next.get_layer_ops()['attrs'][0]['bias']
        if has_bias:
            bias1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][1])

        weight2 = copy.deepcopy(layer_next2.get_layer_ops()['weights'][0])
        weight2_orig = copy.deepcopy(weight2)
        has_bias = layer_next2.get_layer_ops()['attrs'][0]['bias']
        if has_bias:
            bias2 = copy.deepcopy(layer_next2.get_layer_ops()['weights'][1])

        range0 = np.amax(np.abs(weight0), axis=(1, 2, 3))
        range1 = np.amax(np.abs(weight1), axis=(1, 2, 3))
        range2 = np.amax(np.abs(weight2), axis=(0, 2, 3))
        s_01 = range0 / np.power(range0 * range1 * range2, 1.0 / 3)
        s_12 = np.power(range0 * range1 * range2, 1.0 / 3) / range2
        for i in range(len(s_01)):
            weight0[i, :, :, :] /= s_01[i]
            weight1[i, :, :, :] *= (s_01[i] / s_12[i])
            bias0[i] /= s_01[i]# type: ignore
            bias1[i] /= s_12[i]# type: ignore
            weight2[:, i, :, :] *= s_12[i]

        layer.set_layer_ops(dict(weights=[weight0, bias0]))# type: ignore
        layer_next.set_layer_ops(dict(weights=[weight1, bias1]))# type: ignore
        layer_next2.set_layer_ops(dict(weights=[weight2, bias2]))# type: ignore
    elif layer_next and layer_next.get_layer_type() == "batchnormalization":
        for layer_name_ in [layer.get_layer_name(), layer_next.get_layer_name()]:
            if layer_name_ in skip_layer_names:
                return
        print("cross_layer_equalization2: ",
              layer.get_layer_name(), layer_next.get_layer_name())
        weight0 = copy.deepcopy(layer.get_layer_ops()['weights'][0])
        weight0_orig = copy.deepcopy(weight0)
        has_bias = layer.get_layer_ops()['attrs'][0]['bias']
        if has_bias:
            bias0 = copy.deepcopy(layer.get_layer_ops()['weights'][1])

        weight1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][0])
        bias1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][1])
        running_mean = copy.deepcopy(layer_next.get_layer_ops()['weights'][2])
        running_var = copy.deepcopy(layer_next.get_layer_ops()['weights'][3])

        range0 = np.amax(np.abs(weight0), axis=(1,))
        range1 = np.array(np.abs(weight1))
        scale_factor = range0 / np.sqrt(range0 * range1)
        for i in range(len(scale_factor)):
            weight0[i, :] /= scale_factor[i]
            bias0[i] /= scale_factor[i]
            running_mean[i] /= scale_factor[i]
            weight1[i] *= scale_factor[i]

        layer.set_layer_ops(dict(weights=[weight0, bias0]))# type: ignore
        layer_next.set_layer_ops(
            dict(weights=[weight1, bias1, running_mean, running_var]))
        layer_next.set_weight([weight1, bias1, running_mean, running_var])
    elif layer_next and layer.get_layer_type() == "conv" and \
            layer_next.get_layer_type() == "conv" and \
            layer_next2 == None:
        for layer_name_ in [layer.get_layer_name(), layer_next.get_layer_name()]:
            if layer_name_ in skip_layer_names:
                return
        print("cross_layer_equalization2: ",
              layer.get_layer_name(), layer_next.get_layer_name())
        weight0 = copy.deepcopy(layer.get_layer_ops()['weights'][0])
        weight0_orig = copy.deepcopy(weight0)
        has_bias = layer.get_layer_ops()['attrs'][0]['bias']
        if has_bias:
            bias0 = copy.deepcopy(layer.get_layer_ops()['weights'][1])

        weight1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][0])
        weight1_orig = copy.deepcopy(weight1)
        has_bias = layer_next.get_layer_ops()['attrs'][0]['bias']
        if has_bias:
            bias1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][1])

        range0 = np.amax(np.abs(weight0), axis=(1, 2, 3))
        range1 = np.amax(np.abs(weight1), axis=(0, 2, 3))
        scale_factor = range0 / np.sqrt(range0 * range1)
        for i in range(len(scale_factor)):
            weight0[i, :, :, :] /= scale_factor[i]
            bias0[i] /= scale_factor[i]# type: ignore
            if layer_next.get_layer_type() == "depthwiseconv":
                weight1[i, :, :, :] *= scale_factor[i]
            else:
                weight1[:, i, :, :] *= scale_factor[i]

        layer.set_layer_ops(dict(weights=[weight0, bias0]))# type: ignore
        layer_next.set_layer_ops(dict(weights=[weight1, bias1]))# type: ignore
    elif layer_next and layer.get_layer_type() == "fc" and \
        layer_next.get_layer_type() == "fc" and \
        layer_next2 == None:
        for layer_name_ in [layer.get_layer_name(), layer_next.get_layer_name()]:
            if layer_name_ in skip_layer_names:
                return
        print("cross_layer_equalization2: ",
              layer.get_layer_name(), layer_next.get_layer_name())
        weight0 = copy.deepcopy(layer.get_layer_ops()['weights'][0])
        weight0_orig = copy.deepcopy(weight0)
        has_bias = layer.get_layer_ops()['attrs'][0]['bias']
        if has_bias:
            bias0 = copy.deepcopy(layer.get_layer_ops()['weights'][1])

        weight1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][0])
        weight1_orig = copy.deepcopy(weight1)
        has_bias = layer_next.get_layer_ops()['attrs'][0]['bias']
        if has_bias:
            bias1 = copy.deepcopy(layer_next.get_layer_ops()['weights'][1])

        range0 = np.amax(np.abs(weight0), axis=(1))
        range1 = np.amax(np.abs(weight1), axis=(0))
        scale_factor = range0 / np.sqrt(range0 * range1)
        for i in range(len(scale_factor)):
            weight0[i, :] /= scale_factor[i]
            bias0[i] /= scale_factor[i]# type: ignore
            weight1[:, i] *= scale_factor[i]

        layer.set_layer_ops(dict(weights=[weight0, bias0]))# type: ignore
        layer_next.set_layer_ops(dict(weights=[weight1, bias1]))# type: ignore

if __name__ == "__main__":
    pass
