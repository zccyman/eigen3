import copy
import numpy as np


try:
    from export.v1.npu_layer import *
    from utils import Object, Registry, to_bytes, from_bytes, find_key_by_value_in_enum
except Exception:
    from onnx_converter.export.v1.npu_layer import * # type: ignore
    from onnx_converter.utils import Object, Registry, to_bytes, from_bytes, find_key_by_value_in_enum  # type: ignore


QUANT_TYPE: Registry = Registry("quant_type", scope="")
LAYER_TYPE: Registry = Registry("layer_type", scope="")
LAYER_WEIGHT: Registry = Registry("layer_weight", scope="")
     
               
def calc_align(size):
    if size % 4 > 0:
        return size + (4 - size % 4)
    else:
        return size
                    
@QUANT_TYPE.register_module(name="QUANT_NONE")
class QUANT_NONE(object):

    def __init__(self):
        super(QUANT_NONE, self).__init__()
        self.is_align = True
    
        
    @staticmethod
    def _align(ptr, pre_mem_size):
        pre_mem_size = (pre_mem_size + 1 - ptr)
        ptr += calc_align(pre_mem_size) - 1
        return ptr
    
    
    def set_align(self, is_align):
        self.is_align = is_align
        
        
    def __call__(self, contents, ptr):
        params = dict()

        pre_mem_size = copy.deepcopy(ptr)
        qio_type = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.uint8)[0] # type: ignore  
        params["qio_type"] = find_key_by_value_in_enum(qio_type, NpuType_t)     
        pre_mem_size += 1

        if self.is_align:
            ptr = self._align(ptr, pre_mem_size)
                
        return params, ptr


@QUANT_TYPE.register_module(name="QUANT_LUT8_FP")
class QUANT_LUT8_FP(QUANT_NONE):

    def __init__(self):
        super(QUANT_LUT8_FP, self).__init__()

    def __call__(self, contents, ptr):
        params = dict()

        pre_mem_size = copy.deepcopy(ptr)
        qio_type = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.uint8)[0] # type: ignore  
        params["qio_type"] = find_key_by_value_in_enum(qio_type, NpuType_t)     
        pre_mem_size += 1
        params["shift"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],
            dtype=np.int8)[0] # type: ignore
        pre_mem_size += 1
        
        ### reserved
        pre_mem_size += 1
                
        params["offset"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.uint32)[0] # type: ignore
        pre_mem_size += 4
                 
        if self.is_align:
            ptr = self._align(ptr, pre_mem_size)
                                                        
        return params, ptr
    
    
@QUANT_TYPE.register_module(name="QUANT_SHIFT")
class QUANT_SHIFT(QUANT_NONE):

    def __init__(self):
        super(QUANT_SHIFT, self).__init__()

    def __call__(self, contents, ptr):
        params = dict()

        pre_mem_size = copy.deepcopy(ptr)
        qio_type = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.uint8)[0] # type: ignore  
        params["qio_type"] = find_key_by_value_in_enum(qio_type, NpuType_t)     
        pre_mem_size += 1
        params["shift"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],
            dtype=np.int8)[0] # type: ignore
        pre_mem_size += 1
                
        if self.is_align:
            ptr = self._align(ptr, pre_mem_size)
                                            
        return params, ptr
    
    
@QUANT_TYPE.register_module(name="QUANT_SHIFT_ASY")
class QUANT_SHIFT_ASY(QUANT_NONE):

    def __init__(self):
        super(QUANT_SHIFT_ASY, self).__init__()

    def __call__(self, contents, ptr):
        params = dict()

        pre_mem_size = copy.deepcopy(ptr)
        qio_type = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.uint8)[0] # type: ignore  
        params["qio_type"] = find_key_by_value_in_enum(qio_type, NpuType_t)     
        pre_mem_size += 1
        params["shift"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],
            dtype=np.int8)[0] # type: ignore
        pre_mem_size += 1
        reserved = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],
            dtype=np.int8)[0] # type: ignore
        pre_mem_size += 1
        params["in_zero"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.int32)[0] # type: ignore
        pre_mem_size += 4
        params["out_zero"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.int32)[0] # type: ignore
        pre_mem_size += 4
                                        
        ptr += (pre_mem_size - ptr)
                                            
        return params, ptr
    
        
@QUANT_TYPE.register_module(name="QUANT_QUANT")
class QUANT_QUANT(QUANT_NONE):

    def __init__(self):
        super(QUANT_QUANT, self).__init__()

    def __call__(self, contents, ptr):
        params = dict()

        pre_mem_size = copy.deepcopy(ptr)
        qio_type = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.uint8)[0] # type: ignore  
        params["qio_type"] = find_key_by_value_in_enum(qio_type, NpuType_t)     
        pre_mem_size += 1
        reserved = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 2],
            dtype=np.uint8) # type: ignore
        pre_mem_size += 2
        params["scale"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.float32)[0] # type: ignore
        pre_mem_size += 4
        
        ptr += (pre_mem_size - ptr)
                                
        return params, ptr
    
    
@QUANT_TYPE.register_module(name="QUANT_QUANT_ASY")
class QUANT_QUANT_ASY(QUANT_QUANT):

    def __init__(self):
        super(QUANT_QUANT_ASY, self).__init__()

    def __call__(self, contents, ptr):
        params, ptr = super().__call__(contents, ptr)
        pre_mem_size = copy.deepcopy(ptr)
        
        params["zero"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.int32)[0] # type: ignore
        pre_mem_size += 4
               
        ptr += (pre_mem_size - ptr)
                                     
        return params, ptr
   
            
@QUANT_TYPE.register_module(name="QUANT_DEQUANT")
class QUANT_DEQUANT(QUANT_QUANT):

    def __init__(self):
        super(QUANT_DEQUANT, self).__init__()
    
    
@QUANT_TYPE.register_module(name="QUANT_DEQUANT_ASY")
class QUANT_DEQUANT_ASY(QUANT_QUANT_ASY):

    def __init__(self):
        super(QUANT_DEQUANT_ASY, self).__init__()
        
            
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_QUANT")
class QUANT_PER_CHN_QUANT(QUANT_NONE):

    def __init__(self):
        super(QUANT_PER_CHN_QUANT, self).__init__()

    def __call__(self, contents, ptr):
        params = dict()

        pre_mem_size = copy.deepcopy(ptr)
        qio_type = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.uint8)[0] # type: ignore  
        params["qio_type"] = find_key_by_value_in_enum(qio_type, NpuType_t) # type: ignore      
        pre_mem_size += 1
        reserved = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 2],  
            dtype=np.uint8) # type: ignore
        pre_mem_size += 2
        params["offset"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.uint32)[0] # type: ignore
        pre_mem_size += 4

        ptr += (pre_mem_size - ptr)
        
        return params, ptr
    
    
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_QUANT_ASY")
class QUANT_PER_CHN_QUANT_ASY(QUANT_PER_CHN_QUANT):

    def __init__(self):
        super(QUANT_PER_CHN_QUANT_ASY, self).__init__()
        
            
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_DEQUANT")
class QUANT_PER_CHN_DEQUANT(QUANT_PER_CHN_QUANT):

    def __init__(self):
        super(QUANT_PER_CHN_DEQUANT, self).__init__()
        
       
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_DEQUANT_ASY")
class QUANT_PER_CHN_DEQUANT_ASY(QUANT_PER_CHN_QUANT):

    def __init__(self):
        super(QUANT_PER_CHN_DEQUANT_ASY, self).__init__()
        
                    
@QUANT_TYPE.register_module(name="QUANT_ISCALE")
class QUANT_ISCALE(QUANT_NONE):

    def __init__(self):
        super(QUANT_ISCALE, self).__init__()

    def __call__(self, contents, ptr):
        params = dict()

        pre_mem_size = copy.deepcopy(ptr)
        qio_type = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.uint8)[0] # type: ignore  
        params["qio_type"] = find_key_by_value_in_enum(qio_type, NpuType_t) # type: ignore      
        pre_mem_size += 1
        params["shift"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.int8)[0] # type: ignore
        pre_mem_size += 1
        params["scale"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],
            dtype=np.uint8)[0] # type: ignore
        pre_mem_size += 1
        params["s_shift"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],
            dtype=np.int8)[0] # type: ignore
        pre_mem_size += 1

        if self.is_align:
            ptr = self._align(ptr, pre_mem_size)
        
        return params, ptr
    
    
@QUANT_TYPE.register_module(name="QUANT_ISCALE_ASY")
class QUANT_ISCALE_ASY(QUANT_ISCALE):

    def __init__(self):
        super(QUANT_ISCALE_ASY, self).__init__()

    def __call__(self, contents, ptr):
        params, ptr = super().__call__(contents, ptr)
        pre_mem_size = copy.deepcopy(ptr)

        params["in_zero"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.int32) # type: ignore
        pre_mem_size += 4
        params["out_zero"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.int32) # type: ignore
        pre_mem_size += 4
                        
        ptr += (pre_mem_size - ptr)
        
        return params, ptr
    
    
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_ISCALE")
class QUANT_PER_CHN_ISCALE(QUANT_PER_CHN_QUANT):

    def __init__(self):
        super(QUANT_PER_CHN_ISCALE, self).__init__()
    
        
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_ISCALE_ASY")
class QUANT_PER_CHN_ISCALE_ASY(QUANT_PER_CHN_QUANT):

    def __init__(self):
        super(QUANT_PER_CHN_ISCALE_ASY, self).__init__()
        
                
@QUANT_TYPE.register_module(name="QUANT_FSCALE")
class QUANT_FSCALE(QUANT_NONE):

    def __init__(self):
        super(QUANT_FSCALE, self).__init__()

    def __call__(self, contents, ptr):
        params = dict()

        pre_mem_size = copy.deepcopy(ptr)
        qio_type = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.uint8)[0] # type: ignore  
        params["qio_type"] = find_key_by_value_in_enum(qio_type, NpuType_t) # type: ignore      
        pre_mem_size += 1
        params["shift"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],  
            dtype=np.int8)[0] # type: ignore
        pre_mem_size += 1
        reserved = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 1],
            dtype=np.uint8)[0] # type: ignore
        pre_mem_size += 1
        params["scale"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.float32)[0] # type: ignore
        pre_mem_size += 4

        ptr += (pre_mem_size - ptr)
        
        return params, ptr
    

@QUANT_TYPE.register_module(name="QUANT_FSCALE_ASY")
class QUANT_FSCALE_ASY(QUANT_FSCALE):

    def __init__(self):
        super(QUANT_FSCALE_ASY, self).__init__()

    def __call__(self, contents, ptr):
        params, ptr = super().__call__(contents, ptr)
        pre_mem_size = copy.deepcopy(ptr)

        params["in_zero"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.int32) # type: ignore
        pre_mem_size += 4
        params["out_zero"] = from_bytes( # type: ignore
            contents[pre_mem_size:pre_mem_size + 4],
            dtype=np.int32) # type: ignore
        pre_mem_size += 4
                        
        ptr += (pre_mem_size - ptr)
        
        return params, ptr
    
    
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_FSCALE")
class QUANT_PER_CHN_FSCALE(QUANT_PER_CHN_QUANT):

    def __init__(self):
        super(QUANT_PER_CHN_FSCALE, self).__init__()
        

@QUANT_TYPE.register_module(name="QUANT_PER_CHN_FSCALE_ASY")
class QUANT_PER_CHN_FSCALE_ASY(QUANT_PER_CHN_QUANT):

    def __init__(self):
        super(QUANT_PER_CHN_FSCALE_ASY, self).__init__()                     
       
       
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_SHIFT")
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_LUT8_FP")
class QUANT_PER_CHN_SHIFT(QUANT_PER_CHN_QUANT):

    def __init__(self):
        super(QUANT_PER_CHN_SHIFT, self).__init__()
        
             
@QUANT_TYPE.register_module(name="QUANT_PER_CHN_SHIFT_ASY")
class QUANT_PER_CHN_SHIFT_ASY(QUANT_PER_CHN_QUANT):

    def __init__(self):
        super(QUANT_PER_CHN_SHIFT_ASY, self).__init__()
        
                                                
@LAYER_TYPE.register_module(name="LAYER_ACTIVATION")
class Activation(object):

    def __init__(self, isolated=True):
        super(Activation, self).__init__()
        self.isolated = isolated
        
    def __call__(self, contents, ptr):
        params = dict()
        
        if self.isolated:
            format_dict = {
                'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
                'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
                'N': np.uint16, 'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
                'real_c': np.uint16, 
            }
            for key, dtype in format_dict.items():
                value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
                if key in ["i_type", "o_type"]:
                    value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
                elif key in ["i_fmt", "o_fmt"]:
                    value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
                params[key] = value
                ptr += dtype().itemsize
                    
            reserved = from_bytes(contents[ptr:ptr + 2], dtype=np.uint16)[0] # type: ignore
            ptr += 2
        
        act_type = from_bytes(contents[ptr:ptr + 1], dtype=np.uint8)[0] # type: ignore
        ptr += 1
        act_type = find_key_by_value_in_enum(act_type, Activation_t) # type: ignore
        
        if act_type in [
            "ACT_NONE", "ACT_RELU", "ACT_RELU6",
            "ACT_SIGMOID", "ACT_SWISH", "ACT_TANH", "ACT_HARD_SIGMOID",
            "ACT_HARD_SWISH", "ACT_LEAKY_RELU", "ACT_BRELU",
            "ACT_LUT8", "ACT_LUT16",
        ]:
            reserved = from_bytes(contents[ptr:ptr + 3], dtype=np.uint8) # type: ignore
            ptr += 3
            if act_type in ["ACT_HARD_SIGMOID"]:
                params["alpha"], params["beta"] = from_bytes(contents[ptr:ptr + 8], dtype=np.float32) # type: ignore
                ptr += 8
            elif act_type in [
                "ACT_LEAKY_RELU", "ACT_BRELU",
                "ACT_LUT8", "ACT_LUT16",
            ]:
                if act_type == "ACT_LEAKY_RELU":
                    params["alpha"] = from_bytes(contents[ptr:ptr + 4], dtype=np.float32)[0] # type: ignore
                elif act_type == "ACT_BRELU":
                    params["bound"] = from_bytes(contents[ptr:ptr + 4], dtype=np.float32)[0] # type: ignore
                elif act_type in ["ACT_LUT8", "ACT_LUT16"]:   
                    params["offset"] = from_bytes(contents[ptr:ptr + 4], dtype=np.uint32)[0] # type: ignore
                ptr += 4  
                
        params["act_type"] = act_type
                             
        return params, ptr
        
        
@LAYER_TYPE.register_module(name="LAYER_PLACEHOLDER")
class PLACEHOLDER(object):

    def __init__(self):
        super(PLACEHOLDER, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'N': np.uint16, 'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
                
        return params, ptr
    
            
@LAYER_TYPE.register_module(name="LAYER_CONV2D")
@LAYER_TYPE.register_module(name="LAYER_DW_CONV2D")
@LAYER_TYPE.register_module(name="LAYER_TS_CONV2D")
class Conv2d(object):

    def __init__(self):
        super(Conv2d, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'w_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'w_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'H': np.uint16, 'W': np.uint16, 'C': np.uint16, 'FH': np.uint16, # type: ignore
            'FW': np.uint16, 'K': np.uint16, 'SH': np.uint16, 'SW': np.uint16, # type: ignore
            'pad_t': np.uint16, 'pad_b': np.uint16, 'pad_l': np.uint16, # type: ignore
            'pad_r': np.uint16, 'OH': np.uint16, 'OW': np.uint16, 'has_bias': np.uint8, # type: ignore
            'split_chn': np.uint8, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "w_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "w_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
        
        params["w_offset"] = from_bytes(contents[ptr:ptr + 4], dtype=np.uint32)[0] # type: ignore
        ptr += 4       
        
        activation = LAYER_TYPE.get("LAYER_ACTIVATION")(isolated=False) # type: ignore
        params_act, ptr = activation(contents, ptr)
        params["activation"] = params_act
        
        return params, ptr
    
    
@LAYER_TYPE.register_module(name="LAYER_FC")
class FC(object):

    def __init__(self):
        super(FC, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'w_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'w_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'M': np.uint16, 'K': np.uint16, 'N': np.uint16, # type: ignore
            'has_bias': np.uint8, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "w_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "w_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
        
        reserved = from_bytes(contents[ptr:ptr + 3], dtype=np.uint8) # type: ignore
        ptr += 3
        params["w_offset"] = from_bytes(contents[ptr:ptr + 4], dtype=np.uint32)[0] # type: ignore
        ptr += 4       
        
        activation = LAYER_TYPE.get("LAYER_ACTIVATION")(isolated=False) # type: ignore
        params_act, ptr = activation(contents, ptr)
        params["activation"] = params_act
        
        return params, ptr
    
    
@LAYER_TYPE.register_module(name="LAYER_MATMUL")
class MatMul(object):

    def __init__(self):
        super(MatMul, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type0': np.uint8, 'i_type1': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt0': np.uint8, 'i_fmt1': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'N0': np.uint16, 'H0': np.uint16, 'W0': np.uint16, 'C0': np.uint16, # type: ignore
            'N1': np.uint16, 'H1': np.uint16, 'W1': np.uint16, 'C1': np.uint16, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type0", "i_type1", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt0", "i_fmt1", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
                     
        # reserved, np.uint16
        ptr += 2
                                                                    
        return params, ptr
    
        
@LAYER_TYPE.register_module(name="LAYER_RESIZE")
class Resize(object):

    def __init__(self):
        super(Resize, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'IH': np.uint16, 'IW': np.uint16, 'IC': np.uint16, # type: ignore
            'OH': np.uint16, 'OW': np.uint16, 'OC': np.uint16,# type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
        
        method = from_bytes(contents[ptr:ptr + 1], dtype=np.uint8)[0] # type: ignore
        ptr += 1    
        params["method"] = find_key_by_value_in_enum(method, ResizeMethod_t)
        mode = from_bytes(contents[ptr:ptr + 1], dtype=np.uint8)[0] # type: ignore
        ptr += 1          
        params["mode"] = find_key_by_value_in_enum(mode, ResizeCoordinateTransMode_t)
        if params["method"] in ["RESIZE_NEAREST"]:
            round_mode = from_bytes(contents[ptr:ptr + 1], dtype=np.uint8)[0] # type: ignore
            ptr += 1    
            params["round_mode"] = find_key_by_value_in_enum(round_mode, ResizeNearestRoundMode_t)
        elif params["method"] in ["RESIZE_BILINEAR"]:
            reserved = from_bytes(contents[ptr:ptr + 1], dtype=np.uint8)[0] # type: ignore
            ptr += 1  
        else:
            raise NotImplemented
        reserved = from_bytes(contents[ptr:ptr + 1], dtype=np.uint8)[0] # type: ignore
        ptr += 1  
        params["exclude_outside"] = from_bytes(contents[ptr:ptr + 4], dtype=np.int32)[0] # type: ignore
        ptr += 4 
        params["extrapolation_value"] = from_bytes(contents[ptr:ptr + 4], dtype=np.float32)[0] # type: ignore
        ptr += 4  
                                                                 
        return params, ptr
    
        
@LAYER_TYPE.register_module(name="LAYER_NORM")
class Norm(object):

    def __init__(self):
        super(Norm, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'N': np.uint16, 'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
            "operation": np.uint8, "affine": np.uint8, # type: ignore
            "groups": np.uint16, # type: ignore
            "eps": np.float32, # type: ignore
            "affine_offset": np.uint32, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            elif key in ["operation"]:
                value = find_key_by_value_in_enum(value, NormType_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
                                                                         
        return params, ptr
    
            
@LAYER_TYPE.register_module(name="LAYER_REDUCE")
class Reduce(object):

    def __init__(self):
        super(Reduce, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'N': np.uint16, 'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
            'operation': np.uint8, "axis": np.uint8, "keepdims": np.uint8, "reserved": np.uint8, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            elif key in ["operation"]:
                value = find_key_by_value_in_enum(value, ReduceType_t) # type: ignore
            elif key in ["axis"]:
                value = find_key_by_value_in_enum(value, DimType_t) # type: ignore
                
            params[key] = value
            ptr += dtype().itemsize
                                                                         
        return params, ptr
    
    
@LAYER_TYPE.register_module(name="LAYER_TRANSPOSE")
class Transpose(object):

    def __init__(self):
        super(Transpose, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'N': np.uint16, 'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
            "real_c": np.uint16, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
              
        reserved = from_bytes(contents[ptr:ptr + 2], dtype=np.uint16) # type: ignore
        ptr += 2 
                             
        params["perm"] = from_bytes(contents[ptr:ptr + 4], dtype=np.uint8) # type: ignore
        ptr += 4 
                                                                    
        return params, ptr
    
    
@LAYER_TYPE.register_module(name="LAYER_RESHAPE")
class Reshape(object):

    def __init__(self):
        super(Reshape, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'N': np.uint16, 'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
            'ON': np.uint16, 'OH': np.uint16, 'OW': np.uint16, 'OC': np.uint16, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
                                                                                         
        return params, ptr
    
            
@LAYER_TYPE.register_module(name="LAYER_PAD")
class Pad(object):

    def __init__(self):
        super(Pad, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'N': np.uint16, 'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
            'pads': np.uint16, # type: ignore
        }
        for key, dtype in format_dict.items():
            if key in ["pads"]:
                value = from_bytes(contents[ptr:ptr + 8 * dtype().itemsize], dtype=dtype)
                params[key] = list(value) # type: ignore
                ptr += 8 * dtype().itemsize
            else:
                value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
                if key in ["i_type", "o_type"]:
                    value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
                elif key in ["i_fmt", "o_fmt"]:
                    value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
                params[key] = value
                ptr += dtype().itemsize
                                                                                         
        return params, ptr
    
                        
@LAYER_TYPE.register_module(name="LAYER_POOL")
class Pooling(object):

    def __init__(self):
        super(Pooling, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'pool_type': np.uint8, 'scale': np.uint8, # type: ignore
            'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
            'FH': np.uint16, 'FW': np.uint16, # type: ignore
            'SH': np.uint16, 'SW': np.uint16, # type: ignore
            'pad_t': np.uint16, 'pad_b': np.uint16, # type: ignore
            'pad_l': np.uint16, 'pad_r': np.uint16, # type: ignore
            'OH': np.uint16, 'OW': np.uint16, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            elif key in ["pool_type"]:
                value = find_key_by_value_in_enum(value, Pool_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
                                                                                 
        return params, ptr
        
                
@LAYER_TYPE.register_module(name="LAYER_EWS")
class ElementWise(object):

    def __init__(self):
        super(ElementWise, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
        
        operation = from_bytes(contents[ptr:ptr + 1], dtype=np.uint8)[0] # type: ignore
        ptr += 1  
        params["operation"] = find_key_by_value_in_enum(operation, EleWiseType_t) # type: ignore
                
        reserved = from_bytes(contents[ptr:ptr + 1], dtype=np.uint8)[0] # type: ignore
        ptr += 1   
        params["input_len"] = from_bytes(contents[ptr:ptr + 4], dtype=np.uint32)[0] # type: ignore
        ptr += 4
                                                                         
        return params, ptr
    
                
                
@LAYER_TYPE.register_module(name="LAYER_CWS")
class ChannelWise(object):

    def __init__(self):
        super(ChannelWise, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'c_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'c_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            params[key] = value
            ptr += dtype().itemsize
        
        operation = from_bytes(contents[ptr:ptr + 1], dtype=np.uint8)[0] # type: ignore
        ptr += 1  
        params["operation"] = find_key_by_value_in_enum(operation, ChnWiseType_t) # type: ignore
                
        reserved = from_bytes(contents[ptr:ptr + 3], dtype=np.uint8)[0] # type: ignore
        ptr += 3  
                                                                         
        return params, ptr
    
    
@LAYER_TYPE.register_module(name="LAYER_CONCAT")
class Concat(object):

    def __init__(self):
        super(Concat, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'H': np.uint16, 'W': np.uint16, # type: ignore
            'C': np.uint16, 'real_c': np.uint16, # type: ignore
            'OC': np.uint16, 'real_oc': np.uint16, # type: ignore
        }
        for key, dtype in format_dict.items():
            if key in ["C", "real_c"]:
                value = from_bytes(contents[ptr:ptr + 8 * dtype().itemsize], dtype=dtype)
                params[key] = list(value) # type: ignore
                ptr += 8 * dtype().itemsize                
            else:
                value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
                if key in ["i_type", "o_type"]:
                    value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
                elif key in ["i_fmt", "o_fmt"]:
                    value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
                params[key] = value
                ptr += dtype().itemsize
                                                                                                                             
        return params, ptr
    
                       
@LAYER_TYPE.register_module(name="LAYER_SPLIT")
class Split(object):

    def __init__(self):
        super(Split, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'N': np.uint16, 'H': np.uint16, 'W': np.uint16, 'C': np.uint16, # type: ignore
            'operation': np.uint8, 'axis': np.uint8, # type: ignore
            'num': np.uint16, 'pad_value': np.uint32, # type: ignore
        }
        for key, dtype in format_dict.items():
            value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
            if key in ["i_type", "o_type"]:
                value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
            elif key in ["i_fmt", "o_fmt"]:
                value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
            elif key in ["operation"]:
                value = find_key_by_value_in_enum(value, SplitType_t)
            params[key] = value
            ptr += dtype().itemsize
            
        params["split"] = from_bytes(contents[ptr:ptr + 2 * params["num"]], dtype=np.uint16) # type: ignore
        ptr += 2 * params["num"] 
                                                                                                         
        return params, ptr
    

@LAYER_TYPE.register_module(name="LAYER_SHUFFLE")
class Shuffle(object):

    def __init__(self):
        super(Shuffle, self).__init__()
        
    def __call__(self, contents, ptr):
        params = dict()
        format_dict = {
            'i_type': np.uint8, 'o_type': np.uint8, # type: ignore
            'i_fmt': np.uint8, 'o_fmt': np.uint8, # type: ignore
            'N': np.uint16, # type: ignore
            'H': np.uint16, 'W': np.uint16, # type: ignore
            'C': np.uint16, 'real_c': np.uint16, # type: ignore
            'OC': np.uint16, 'real_oc': np.uint16, # type: ignore
            'axis': np.uint8, 'segments': np.uint8, # type: ignore
        }
        for key, dtype in format_dict.items():
            if key in ["C", "real_c", "OC", "real_oc"]:
                value = from_bytes(contents[ptr:ptr + 4 * dtype().itemsize], dtype=dtype)
                params[key] = list(value) # type: ignore
                ptr += 4 * dtype().itemsize                
            else:
                value = from_bytes(contents[ptr:ptr + dtype().itemsize], dtype=dtype)[0] # type: ignore
                if key in ["i_type", "o_type"]:
                    value = find_key_by_value_in_enum(value, NpuType_t) # type: ignore
                elif key in ["i_fmt", "o_fmt"]:
                    value = find_key_by_value_in_enum(value, LayerFmt_t) # type: ignore
                params[key] = value
                ptr += dtype().itemsize
                                                                                                                             
        return params, ptr
    
                                
@LAYER_WEIGHT.register_module(name="LAYER_CONV2D")
@LAYER_WEIGHT.register_module(name="LAYER_DW_CONV2D")
class Conv2d_Weight(object):

    def __init__(self, data_channel_extension=False, first_conv=False):
        super(Conv2d_Weight, self).__init__()
        self.data_channel_extension = data_channel_extension
        self.first_conv = first_conv
        
    def __call__(self, params, contents, ptr, layer_type):
        weight_dict = dict()
        
        ptr_cpy = copy.deepcopy(ptr)
        
        K = params["K"]
        C = params["C"]
        if self.data_channel_extension and self.first_conv:
            C *= 4
        FH = params["FH"]
        FW = params["FW"]
        w_type = params["w_type"]
        if w_type in ["NPU_INT8", "NPU_UINT8"]:
            dtype = np.int8
            bias_type = np.int32
            weight_size = K * C * FH * FW # type: ignore
            if layer_type in ["depthwiseconv"]:
                bias_size = C * 4
            else:
                bias_size = K * 4
        else:
            dtype = np.int16
            bias_type = np.int64
            weight_size = K * C * FH * FW * 2 # type: ignore
            if layer_type in ["depthwiseconv"]:
                bias_size = C * 8
            else:
                bias_size = K * 8
            
        ptr_cpy += params["w_offset"]
        weight_dict["weight"] = np.array(from_bytes(contents[ptr_cpy:ptr_cpy+weight_size], dtype=dtype)) # type: ignore
        ptr_cpy += weight_size
        
        has_bias = params["has_bias"]
        if has_bias:
           weight_dict["bias"] = np.array(from_bytes(contents[ptr_cpy:ptr_cpy+bias_size], dtype=bias_type)) # type: ignore
           ptr_cpy += bias_size

        if "offset" in params["activation"]:
            ptr_cpy = copy.deepcopy(ptr)
            ptr_cpy += params["activation"]["offset"]
            if params["o_type"] in ["NPU_INT8"]:
                dtype = np.int8
                table_size = 2 ** 8
                weight_dict["act_table"] = np.array(from_bytes(contents[ptr_cpy:ptr_cpy+table_size], dtype=dtype)) # type: ignore
                ptr_cpy += table_size
            else:
                dtype = np.int16
                table_size = 2 ** 16
                weight_dict["act_table"] = np.array(from_bytes(contents[ptr_cpy:ptr_cpy+table_size], dtype=dtype)) # type: ignore
                ptr_cpy += table_size
        
        return weight_dict
    

@LAYER_WEIGHT.register_module(name="LAYER_FC")
class Fc_Weight(object):

    def __init__(self, data_channel_extension=False, first_conv=False):
        super(Fc_Weight, self).__init__()
        self.data_channel_extension = data_channel_extension
        self.first_conv = first_conv
        
    def __call__(self, params, contents, ptr, layer_type):
        weight_dict = dict()
        
        ptr_cpy = copy.deepcopy(ptr)
        
        M = params["M"]
        K = params["K"]
        N = params["N"]

        w_type = params["w_type"]
        if w_type in ["NPU_INT8", "NPU_UINT8"]:
            dtype = np.int8
            bias_type = np.int32
            weight_size = K * N # type: ignore
            bias_size = N * 4
        else:
            dtype = np.int16
            bias_type = np.int64
            weight_size = K * N * 2 # type: ignore
            bias_size = N * 4
            
        ptr_cpy += params["w_offset"]
        weight_dict["weight"] = np.array(from_bytes(contents[ptr_cpy:ptr_cpy+weight_size], dtype=dtype)) # type: ignore
        ptr_cpy += weight_size
        
        has_bias = params["has_bias"]
        if has_bias:
           weight_dict["bias"] = np.array(from_bytes(contents[ptr_cpy:ptr_cpy+bias_size], dtype=bias_type)) # type: ignore
           ptr_cpy += bias_size

        if "offset" in params["activation"]:
            ptr_cpy = copy.deepcopy(ptr)
            ptr_cpy += params["activation"]["offset"]
            if params["o_type"] in ["NPU_INT8"]:
                dtype = np.int8
                table_size = 2 ** 8
                weight_dict["act_table"] = np.array(from_bytes(contents[ptr_cpy:ptr_cpy+table_size], dtype=dtype)) # type: ignore
                ptr_cpy += table_size
            else:
                dtype = np.int16
                table_size = 2 ** 16
                weight_dict["act_table"] = np.array(from_bytes(contents[ptr_cpy:ptr_cpy+table_size], dtype=dtype)) # type: ignore
                ptr_cpy += table_size
        
        return weight_dict