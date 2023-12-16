# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2021/10/8 17:07
# @File     : data_correct.py

# calc final scale registry
import os
import numpy as np
# from abc import abstractmethod
try:
    from utils import get_scale_shift
    from utils import Registry, extract_scale
except:
    from onnx_converter.utils import get_scale_shift # type: ignore
    from onnx_converter.utils import Registry, extract_scale # type: ignore

DATACORRECT: Registry = Registry(name='correcting', scope='')

zero, one = np.int32(0), np.int32(1)
f_zero, f_one = np.float32(0), np.float32(1)
min_eps = 1e-2

class BaseScale(object):
    def __init__(self):
        super(BaseScale, self).__init__()
        self.eps = 1e-5
        self.si, self.sk, self.so = f_one, f_one, f_one
        self.zi, self.zk, self.zo = zero, zero, zero
        self.out_shift, self.out_scale = zero, one

    def get_class_name(self):
        return self.__class__.__name__

    # c/c++ using struct transfer parameter
    def get_param(self):
        param = dict(si=self.si, sk=self.sk, so=self.so,
                     zi=self.zi, zk=self.zk, zo=self.zo,
                     out_shift=self.out_shift, 
                     out_scale=self.out_scale)
        return param
    
    # c/c++ using struct transfer parameter
    def set_param(self, param: dict):
        self.si = param.get('si', self.si)
        self.sk = param.get('si', self.sk)
        self.so = param.get('si', self.so)
        self.zi = param.get('si', self.zi)
        self.zk = param.get('si', self.zk)
        self.zo = param.get('si', self.zo)
        self.out_shift = param.get('si', self.out_shift)
        self.out_scale = param.get('si', self.out_scale)
    

@DATACORRECT.register_module(name='intscale')
class IntScale(BaseScale):
    # kwargs['bits_dict'], kwargs['maxs'], kwargs['mins'], kwargs['int_scale']
    # kwargs['si'], kwargs['so'], kwargs['sk']

    @staticmethod
    def clip_scale(self): # type: ignore
        self.min_scale = self.eps / self.maxs[self.bit_select]
        self.si = np.clip(self.si, self.min_scale, 1/self.eps)
        self.sk = np.clip(self.sk, self.min_scale, 1/self.eps)
        self.so = np.clip(self.so, self.min_scale, 1/self.eps)

    def __init__(self, **kwargs):
        super(IntScale, self).__init__()
        self.bit_select = kwargs['bit_select']
        self.bits_dict, self.maxs = kwargs['bits_dict'], kwargs['maxs']
        self.mins, self.int_scale = kwargs['mins'], kwargs['int_scale']

        flag = 'si' in kwargs.keys() and 'so' in kwargs.keys()
        if not flag:
            print('not enough quantize parameter!')
            os._exit(-1)
        # self.si, self.so = kwargs['si'], kwargs['so']
        self.si = extract_scale(kwargs['si'])
        self.so = extract_scale(kwargs['so'])
        self.zi = extract_scale(kwargs['si'], 'zero_point', zero) # type: ignore
        self.zo = extract_scale(kwargs['so'], 'zero_point', zero) # type: ignore

        if 'sk' in kwargs.keys():
            if isinstance(kwargs['sk'], list):
                self.sk = [extract_scale(sk['scale']) for sk in kwargs['sk']]
                self.zk = [extract_scale(sk, 'zero_point', zero) for sk in kwargs['sk']] # type: ignore
                if len(self.sk) == 1:
                    self.sk = self.sk[0]
                    self.zk = self.zk[0]
            else:
                self.sk = extract_scale(kwargs['sk']['scale'])
                self.zk = extract_scale(kwargs['sk'], 'zero_point', zero) # type: ignore
        else:
            self.sk, self.zk = one, zero
        self.clip_scale(self)
    
    def process_zero_point(self):
        self.zo /= (self.si * self.sk / self.so) # type: ignore

        # self.zero_points = [0, 0, 0]
        # self.zero_points = kwargs['zero_points'] if 'zero_points' in kwargs.keys() else self.zero_points

    def update_quantize(self, **kwargs):
        self.si = extract_scale(kwargs.get('si', self.si))
        self.sk = extract_scale(kwargs.get('sk', self.sk))
        self.so = extract_scale(kwargs.get('so', self.so))
        return self.update_scale()

    def update_scale(self):
        out_shift, out_scale = get_scale_shift(self.si * self.sk / self.so) # type: ignore
        out_scale = np.int32(out_scale * (2 ** self.int_scale))
        out_scale = np.clip(out_scale, 0, (2**self.int_scale) - 1)
        return dict(out_shift=out_shift, out_scale=out_scale, extra_value=zero,
                    zi=self.zi, zk=self.zk, zo=self.zo)

    def update_bit(self, bit_select):
        self.bit_select = bit_select

    def __call__(self):
        try:
            return self.update_scale()
        except:
            error_info = "method {} quantize alignment error".format(self.get_class_name())
            print(error_info)
            os._exit(-1)

    def forward(self, in_data):
        pass


@DATACORRECT.register_module(name='intscaleex')
class IntScaleEx(BaseScale):
    # kwargs['bits_dict'], kwargs['maxs'], kwargs['mins'], kwargs['int_scale']
    # kwargs['si'], kwargs['so'], kwargs['sk']

    @staticmethod
    def clip_scale(self): # type: ignore
        self.min_scale = self.eps / self.maxs[self.bit_select]
        self.si = np.clip(self.si, self.min_scale, 1/self.eps)
        self.sk = np.clip(self.sk, self.min_scale, 1/self.eps)
        self.so = np.clip(self.so, self.min_scale, 1/self.eps)

    def __init__(self, **kwargs):
        super(IntScaleEx, self).__init__()
        self.bit_select = kwargs['bit_select']
        self.bits_dict, self.maxs = kwargs['bits_dict'], kwargs['maxs']
        self.mins, self.int_scale = kwargs['mins'], kwargs['int_scale']

        flag = 'si' in kwargs.keys() and 'so' in kwargs.keys()
        if not flag:
            print('not enough quantize parameter!')
            os._exit(-1)
        # self.si, self.so = kwargs['si'], kwargs['so']
        self.si = extract_scale(kwargs['si'])
        self.so = extract_scale(kwargs['so'])
        self.zi = extract_scale(kwargs['si'], 'zero_point', zero) # type: ignore
        self.zo = extract_scale(kwargs['so'], 'zero_point', zero) # type: ignore

        if 'sk' in kwargs.keys():
            if isinstance(kwargs['sk'], list):
                self.sk = [extract_scale(sk['scale']) for sk in kwargs['sk']]
                self.zk = [extract_scale(sk, 'zero_point', zero) for sk in kwargs['sk']] # type: ignore
                if len(self.sk) == 1:
                    self.sk = self.sk[0]
                    self.zk = self.zk[0]
            else:
                self.sk = extract_scale(kwargs['sk']['scale'])
                self.zk = extract_scale(kwargs['sk'], 'zero_point', zero) # type: ignore
        else:
            self.sk, self.zk = one, zero
        self.clip_scale(self)

    def process_zero_point(self):
        self.zo /= (self.si * self.sk / self.so) # type: ignore

        # self.zero_points = [0, 0, 0]
        # self.zero_points = kwargs['zero_points'] if 'zero_points' in kwargs.keys() else self.zero_points

    def update_quantize(self, **kwargs):
        self.si = kwargs['si'] if 'si' in kwargs['si'] else self.si
        self.sk = kwargs['si'] if 'sk' in kwargs['sk'] else self.sk
        self.so = kwargs['so'] if 'so' in kwargs['so'] else self.so
        self.update_scale()

    def update_scale(self):
        out_shift, out_scale = get_scale_shift(self.si * self.sk / self.so) # type: ignore
        out_scale = np.int32(out_scale * (2 ** self.int_scale))
        scale = np.clip(out_scale, 0, (2**self.int_scale) - 1)
        extra_value = zero
        
        extra_value = np.int32(np.round(2**(self.int_scale) / scale / 2))
            
        return dict(out_shift=out_shift, out_scale=out_scale, 
                    int_scale=self.int_scale-1,extra_value=extra_value,
                    zi=self.zi, zk=self.zk, zo=self.zo)

    def update_bit(self, bit_select):
        self.bit_select = bit_select

    def __call__(self):
        try:
            return self.update_scale()
        except:
            error_info = "method {} quantize alignment error".format(self.get_class_name())
            print(error_info)
            os._exit(-1)

    def forward(self, in_data):
        pass

@DATACORRECT.register_module(name='shiftfloatscale')
class ShiftFloatScale(IntScale):
    def __init__(self, **kwargs):
        super(ShiftFloatScale, self).__init__(**kwargs)

    def update_scale(self):
        out_shift, out_scale = get_scale_shift(self.si * self.sk / self.so) # type: ignore
        return dict(out_shift=out_shift, out_scale=np.float32(out_scale), extra_value=zero,
                    zi=self.zi, zk=self.zk, zo=self.zo)


@DATACORRECT.register_module(name='shiftfloatscaletable')
@DATACORRECT.register_module(name='shiftfloatscaletable2float')
class ShiftFloatScaleTable(IntScale):
    def __init__(self, **kwargs):
        super(ShiftFloatScaleTable, self).__init__(**kwargs)

    def update_scale(self):
        out_shift, out_scale = get_scale_shift(self.si * self.sk / self.so) # type: ignore
        return dict(out_shift=out_shift, out_scale=np.float32(out_scale*self.so), extra_value=zero,
                    fscale=self.so, zi=self.zi, zk=self.zk, zo=self.zo)
    
    # def __call__(self):
    #     return self.update_scale()


# convolution is result, output type is int8/int16
# conv, depthwiseconv, fc layer
@DATACORRECT.register_module(name='rshiftscale')
class RShiftScale(IntScale):
    def __init__(self, **kwargs):
        super(RShiftScale, self).__init__(**kwargs)

    def update_scale(self):
        out_shift, out_scale = get_scale_shift(self.si * self.sk / self.so) # type: ignore
        return dict(out_shift=out_shift, out_scale=np.float32(1), fscale=np.float32(out_scale * self.so),
                    extra_value=zero, zi=self.zi, zk=self.zk, zo=self.zo)


# result layer does not using so correcting
# conv, depthwiseconv, fc layer
@DATACORRECT.register_module(name='rrshiftscale')
class RRShiftScale(IntScale):
    def __init__(self, **kwargs):
        super(RRShiftScale, self).__init__(**kwargs)

    def update_scale(self):
        out_shift, out_scale = get_scale_shift(self.si * self.sk) # type: ignore
        return dict(out_shift=out_shift, out_scale=np.float32(1), fscale=np.float32(out_scale),
                    extra_value=zero,zi=self.zi, zk=self.zk, zo=self.zo)


# pre concatenate int scale
@DATACORRECT.register_module(name='preintscale')
class PreIntScale(IntScale):
    def __init__(self, **kwargs):
        super(PreIntScale, self).__init__(**kwargs)
        self.scale_eps = 1.0#3

    def update_scale(self):
        int_scale = self.int_scale
        max_value = 2 ** self.int_scale - 1#self.maxs[self.bit_select]
        out_scale = np.float32(self.si * self.sk / self.so) # type: ignore
        
        scale = zero
        if np.abs(out_scale - 1) <= min_eps:
            scale = 1
            self.int_scale = 0
        else:
            while True:

                if self.int_scale == int_scale - 1:
                    max_value = 2 ** self.int_scale - 1
                
                if isinstance(out_scale, np.ndarray):
                    break
                
                scale = np.int32(out_scale * (2 ** self.int_scale) * self.scale_eps)
                    
                # scale = np.int32(out_scale * (2 ** self.int_scale) * eps)
                if scale <= max_value:
                    break
                self.int_scale -= 1
            if scale == 0 or self.int_scale < 0:
                self.int_scale = 0
                scale = one
        
        return dict(out_shift=zero, out_scale=scale, int_scale=self.int_scale,
                    extra_value=zero, zi=self.zi, zk=self.zk, zo=self.zo)


# pre concatenate int scale
@DATACORRECT.register_module(name='preintscaleex')
class PreIntScaleEx(PreIntScale):
    def __init__(self, **kwargs):
        super(PreIntScaleEx, self).__init__(**kwargs)
        self.scale_eps = 1.0#3
    
    def update_scale(self):
        int_scale = self.int_scale
        max_value = 2 ** self.int_scale - 1#self.maxs[self.bit_select]
        out_scale = np.float32(self.si * self.sk / self.so) # type: ignore
        
        scale = zero
        if np.abs(out_scale - 1) <= min_eps:
            scale = 1
            self.int_scale = 0
        else:
            while True:

                if self.int_scale == int_scale - 1:
                    max_value = 2 ** self.int_scale - 1
                
                if isinstance(out_scale, np.ndarray):
                    break
                
                scale = np.int32(out_scale * (2 ** self.int_scale) * self.scale_eps)
                    
                # scale = np.int32(out_scale * (2 ** self.int_scale) * eps)
                if scale <= max_value:
                    break
                self.int_scale -= 1
            if scale == 0 or self.int_scale <= 0:
                self.int_scale = zero
                scale = one

        # int_scale = self.int_scale - 1 if self.int_scale > 1 else self.int_scale
        # if self.int_scale < int_scale:
        #     print()
        outputs = dict(out_shift=zero, out_scale=scale, int_scale=self.int_scale, 
                       extra_value=zero, zi=self.zi, zk=self.zk, zo=self.zo)
        if outputs["int_scale"] > 1:
            if self.int_scale in [8, 16]:
                outputs["extra_value"] = 0
            else:
                outputs["int_scale"] -= 1
                outputs["out_scale"] = np.int32(np.round(outputs["out_scale"] / 2))
                outputs["extra_value"] = np.int32(np.round(2**(self.int_scale) / scale / 2))            
            
        return outputs  


@DATACORRECT.register_module(name='floatscale')
class FloatScale(IntScale):
    def __init__(self, **kwargs):
        super(FloatScale, self).__init__(**kwargs)

    def update_scale(self):
        scale = np.float32(self.si * self.sk / self.so) # type: ignore
        out_shift = np.zeros_like(scale, dtype=np.int32) if isinstance(scale, np.ndarray) else 0

        return dict(out_shift=out_shift, out_scale=scale, extra_value=zero, 
                    zi=self.zi, zk=self.zk, zo=self.zo)


@DATACORRECT.register_module(name='ffloatscale')
class FFloatScale(IntScale):
    def __init__(self, **kwargs):
        super(FFloatScale, self).__init__(**kwargs)

    def update_scale(self):
        scale = np.float32(self.si * self.sk) # type: ignore

        return dict(out_shift=zero, out_scale=scale, extra_value=zero,
                    zi=self.zi, zk=self.zk, zo=self.zo)


@DATACORRECT.register_module(name='smooth')
class Smooth(IntScale):
    def __init__(self, **kwargs):
        super(Smooth, self).__init__(**kwargs)

    def __call__(self):
        return dict(out_shift=zero, out_scale=one, extra_value=zero,
                    int_scale=zero, zi=zero, zk=zero, zo=zero)


@DATACORRECT.register_module(name='float')
class Float(IntScale):
    # process de-quantize to float -> process data -> quantize output
    def __init__(self, **kwargs):
        super(Float, self).__init__(**kwargs)
        # self.bit_select = kwargs['bit_select']

    def __call__(self):
        return dict(out_shift=zero, out_scale=one, extra_value=zero,
                    zi=zero, zk=zero, zo=zero)


@DATACORRECT.register_module(name='table')
class Table(IntScale):
    # process de-quantize to float -> process data -> quantize output
    def __init__(self, **kwargs):
        super(Table, self).__init__(**kwargs)
        # self.bit_select = kwargs['bit_select']

    def __call__(self):
        return dict(out_shift=zero, out_scale=one, extra_value=zero,
                    zi=self.zi, zk=self.zk, zo=self.zo)


# @DATACORRECT.register_module(name='intscaleperchannel')
# class IntScalePerChannel(IntScale):
#     def __init__(self, bit_select=1, **kwargs):
#         super(IntScalePerChannel, self).__init__(bit_select, **kwargs)
#
#     def __call__(self, *args, **kwargs):
#         out_shift, out_scale = 0, 0.0
#         return out_shift, out_scale
#
#
# @DATACORRECT.register_module(name='floatscaleperchannel')
# class FloatScalePerchannel(IntScale):
#     def __init__(self, bit_select=1, **kwargs):
#         super(FloatScalePerchannel, self).__init__(bit_select, **kwargs)
#
#     def __call__(self, *args, **kwargs):
#         scale = 0.0
#         return scale


# class BaseMethod(object):
#     def __init__(self, si, sk, so):
#         self.si, self.sk, self.so = si, sk, so
#
#     @abstractmethod
#     def process(self, data):
#         pass
#
#     def __call__(self, data):
#         return self.process(data)
#
#
# @DATACORRECT.register_module(name='BaseInt32')
# class BaseInt32(BaseMethod):
#     def __init__(self, si, sk, so):
#         super(BaseInt32, self).__init__(si, sk, so)
#         self.dtype = np.int32
#
#     def get_dtype(self):
#         return self.dtype
#
#
# @DATACORRECT.register_module(name='Int32toInt8')
# class Int32toInt8(BaseInt32):
#     def __init__(self, si, sk, so, zero_point=0):
#         super(Int32toInt8, self).__init__(si, sk, so)
#         self.name = 'Int32toInt8'
#         self.zero_point = zero_point
#         self.scale = self.si * self.sk / self.so
#
#     def get_type(self):
#         return self.name
#
#     def get_scale(self):
#         return self.scale
#
#     def correct_data(self, data, num_bits, dtype):
#         qmin = -(int)(1 << (num_bits - 1))
#         qmax = (int)((1 << (num_bits - 1)) - 1)
#         transformed_val = data / self.scale + self.zero_point
#         clamped_val = np.clip(transformed_val, qmin, qmax)
#         quantized = np.round(clamped_val)
#         return quantized.astype(dtype)
#
#     def post_layer_process(self, data, num_bits, dtype):
#         data = np.array(data, dtype=self.dtype)
#         return self.correct_data(data, num_bits, dtype)
#
#
# @DATACORRECT.register_module(name='Int32toInt8Intscale')
# class Int32toInt8IntScale(BaseInt32):
#     def __init__(self, si, sk, so):
#         pass
#
#
# @DATACORRECT.register_module(name='Int32toFloat')
# class Int32toFloat(BaseInt32):
#     pass
#
#
# # Int32toInt8PerchannelScale
# @DATACORRECT.register_module(name='Int32toInt8PerChnScale')
# class Int32toInt8PerChnScale(BaseInt32):
#     # Int32toin8PerchannelScaleBeforeBias
#     pass
#
#
# @DATACORRECT.register_module(name='BaseInt8')
# class BaseInt8(BaseMethod):
#     pass
#
#
# @DATACORRECT.register_module(name='Int8toInt8')
# class Int8toInt8(BaseInt8):
#     pass
#
#
# @DATACORRECT.register_module(name='Int8toInt8Intscale')
# class Int8toInt8IntScale(BaseInt8):
#     pass
#
#
# @DATACORRECT.register_module(name='Int8toFloat')
# class Int8toFloat(BaseInt8):
#     pass
#
#
# @DATACORRECT.register_module(name='Int8toInt8PerChnScale')
# class Int8toInt8PerChnScale(BaseInt8):
#     pass


if __name__ == '__main__':
    method = DATACORRECT.get('float')()
    name = method.get_class_name()
    print('test')
