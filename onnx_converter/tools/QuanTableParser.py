import numpy as np
import json, yaml
#from quantizer import graph_quan

conv_like_ops = ['conv', 'depthwiseconv', 'gemm', 'fc']

def parse_NCNN_quant_table(graph, table_file):
    f = open(table_file, "r")
    scale_table = dict()
    weight_scale_dict = dict()
    top_scale_dict = dict()
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.strip('\n')
        line = line.split()
        name = line[0]
        if name.endswith('_param_0'):
            # weight scale info line
            layer_name = name[:name.find("_param")]
            weight_scales = [1/float(x) for x in line[1:]]
            scale_table.update({layer_name:{"weight_scales":weight_scales}})
        else:
            if name.endswith("_top"):
                # top scale
                layer_name = name[:name.find("_top")]
                top_scales = [1/float(x) for x in line[1:]]
                if layer_name in scale_table.keys():
                    scale_table[layer_name].update({"top_scales":top_scales})
                else:
                    scale_table.update({layer_name:{"top_scales":top_scales}})
            else:
                # bottom scale
                #layer_name = name
                #bottom_scales = [1/float(x) for x in line[1:]]
                
                #if layer_name in scale_table.keys():
                #    scale_table[layer_name].update({"bottom_scales":bottom_scales})
                #else:
                #    scale_table.update({layer_name:{"bottom_scales":bottom_scales}})      
                pass

    #weights_dict=dict() # dict , key = layer_idx, value = dict{'weight_scales', 'bottom_scales', 'top_scales'}
    for layer in graph.get_layers():
        #layer_idx = layer.layer_idx
        layer_name = layer.get_layer_name()
        if not layer_name in scale_table.keys():
            print("layer %s not match any in table"%(layer_name))
            continue
        weight_scale_dict.update({layer_name:{}})
        if "weight_scales" in scale_table[layer_name].keys():
            weight_scales = scale_table[layer_name]['weight_scales']
            weight_scale_dict[layer_name].update({'scale':weight_scales,"zero_point":[0]*len(weight_scales), 'bitwidth':8})
        
        #outputs
        # outputs = layer.get_onnx_output_name()
        # for (i,feature_name) in enumerate(outputs):
        #     top_scale_info = scale_table[layer_name]['top_scales']
        #     top_scale_dict.update({feature_name:dict(scale=top_scale_info[i], zero_point=0,bitwidth=8)})
    return weight_scale_dict, None


def parse_AIMET_quant_table(graph, table_file):
    f = open(table_file, "r") 
    if table_file.endswith('json'):
        quan_info = json.load(f)
    if table_file.endswith('yaml'):
        quan_info = yaml.safe_load(f.read())
    f.close()

    dict_weight = quan_info['param_encodings'] # type: ignore
    dict_tensor = quan_info['activation_encodings'] # type: ignore
    weight_scale_dict = None # dict , format {layername :{'scale':scale, 'zero_point':zp, 'bitwidth':bits}} 
    top_scale_dict = None    # dict , format {tensorname:{'scale':scale, 'zero_point':zp, 'bitwidth':bits}} 
    if len(dict_weight.keys()) > 0:
        weight_scale_dict = dict()
    if len(dict_tensor.keys())>0:
        top_scale_dict = dict()

    # weight name mapping
    table_dist_feats=list()  
    quan_params_list = list()
    
    # check if user provide weight std / mean  
    use_data_distribution_match = False
    content = list(dict_weight.values())[0]
    if isinstance(content, list):
        content = content[0]
    if 'mapping' in content.keys() and ['mean', 'std']==content['mapping'].keys():
        use_data_distribution_match = True

    for key in dict_weight.keys():
        if key.endswith('bias'):
            continue
        info = dict_weight[key]
        if isinstance(info, list):
            info = info[0]
        quan_params = {'name':key, 'scale': info['scale'], 'zero_point':info['offset'], 'bitwidth':info['bitwidth'], 'max':info['max'], 'min':info['min']}
        quan_params_list.append(quan_params)
        if use_data_distribution_match:
            dist_feat = np.array([info['mapping']['mean'],info['mapping']['std']])
            table_dist_feats.append(dist_feat)

    if use_data_distribution_match:
        table_dist_feats=np.array(table_dist_feats)
    
    for layer in graph.get_layers():
        layer_name = layer.get_layer_name()
        if layer.get_layer_type().lower() in conv_like_ops:
            weight = layer.get_nodes()[0]._Node__weights[0]
            weight_name = weight['name']
            if use_data_distribution_match:
                weight_data = layer.get_nodes()[0]._Node__weights[0]['weight']
                weight_data = weight_data.flatten()
                mean_val = np.mean(weight_data)
                std_val = np.std(weight_data)
                dist_feat = np.array([mean_val, std_val])
                diff = np.abs(dist_feat - table_dist_feats) # type: ignore
                diff_sum = np.sum(diff, axis=1)  
                idx = np.argmin(diff_sum)
                #print("match idx is ", idx, "diff is " , diff[idx])       
                weight_scale_dict.update({layer_name:quan_params_list[idx]}) # type: ignore
            else: # use weight initializer name for match
                for (idx, quan_params) in enumerate(quan_params_list):
                    if weight_name == quan_params['name']:
                        weight_scale_dict.update({layer_name:quan_params_list[idx]}) # type: ignore


    if len(dict_tensor.keys())>0:
        for layer in graph.get_layers():
            #layer_idx = layer.layer_idx
            layer_name = layer.get_layer_name()
            outputs = layer.get_onnx_output_name()
            for feature_name in outputs:
                if feature_name not in dict_tensor:
                    continue
                top_scale_info = dict_tensor[feature_name]
                max_val = top_scale_info['max']
                min_val = top_scale_info['min']
                if max_val * min_val > 0:
                    max_val = np.maximum(zp, max_val) # type: ignore
                    min_val = np.minimum(zp, min_val) # type: ignore

                # scale= (max_val - min_val) / 255
                # zp = np.round(min_val / scale)
                scale = top_scale_info['scale']
                zp = top_scale_info['offset']        
                bitwidth = top_scale_info['bitwidth']
                top_scale_dict.update({feature_name:{'scale':scale, 'zero_point':zp, 'bitwidth':bitwidth,"max":max_val, "min":min_val}}) # type: ignore

    return weight_scale_dict, top_scale_dict


class QuanTableParser(object):
    def __init__(self, quantizer_name):
        self.offline_quan_support_list = ['AIMET', 'NCNN', 'ONNXRT', 'OPENVIVO']
        assert quantizer_name.upper() in self.offline_quan_support_list, \
        'Specified offline quantization method is not support'
        self.offline_quan_method = quantizer_name
        

    def parse_weight_and_top_scales(self, graph, table_file):
        if self.offline_quan_method == "AIMET":
            weight_scale_dict, top_scale_dict = parse_AIMET_quant_table(graph, table_file)
        if self.offline_quan_method == "NCNN":
            weight_scale_dict, top_scale_dict = parse_NCNN_quant_table(graph, table_file)
        if self.offline_quan_method == "ONNXRT":
            pass
        if self.offline_quan_method == "OPENVIVO":
            pass
        return weight_scale_dict, top_scale_dict # type: ignore


    # def set_graph_weight_scales(self, scale_table):
    #     for layer_ in self.graph.get_layers():
    #         name = layer_.get_layer_name()
    #         if name in scale_table.keys():
    #             #print(layer_)
    #             scale = scale_table[name]['weight_Scales']['scale']
    #             offset = scale_table[name]['weight_Scales']['offset']
    #             layer_.set_w_scale(scale)
    #             layer_.set_w_shift(offset)
    # def layer_name_to_outputs(self):
    #     layer_output_dict=dict()
    #     for layer in self.graph.get_layers():
    #         layer_name = layer.get_layer_name()
    #         outputs = layer.get_onnx_output_name()
    #         layer_output_dict.update({layer_name:outputs})
    #     return layer_output_dict                 
    # def get_featuremap_scales(self, scale_table):
    #     layer_output_dict = self.layer_name_to_outputs()
    #     top_scale_table=dict()
    #     for layer_name in scale_table.keys():
    #         top_list = layer_output_dict[layer_name]
    #         for (idx, feature_name) in enumerate(top_list):
    #             top_scale_info = top_scale = scale_table[layer_name]['top_scales']
    #             scale = top_scale_info['scale'][idx]
    #             zero_point = top_scale_info['zero_point'][idx]
    #             top_scale_table.update({feature_name:dict(scale=scale, zero_point=zero_point)})
    #     return top_scale_table

    # def quantize(self, graph, table_file):
    #     #self.layers = graph.get_layers()
    #     self.graph = graph
    #     self.graph.push_virtual_act()
    #     self.parse_scale_table(table_file)
    #     self.set_weight_scales(self.weight_scale_dict)
    #     self.graph.quan_feats(self.top_scale_dict)
    #     self.graph.quan_weights()
    #     self.graph.quan_ops()
    #     self.graph.__is_quantized = True
    #     print("model quantize done")
