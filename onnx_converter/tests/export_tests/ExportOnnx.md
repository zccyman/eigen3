# ExportOnnx

## Introduction

- All cell states and hidden states in GRU/LSTM must be input as input nodes, and are not allowed to be stored in the init_h and init_c fields of the GRU/LSTM node.

## pytorch2onnx

### GRU

```
import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(257, 128, 1)
  
    def forward(self, input, h0):
        output, (hn) = self.rnn(input, (h0))
  
        return [output, hn]
  
model = GRU()
h0 = torch.randn(1, 1, 128)
input = torch.randn(1, 1, 257)
output, hn = model(input, h0)

torch.onnx.export(model,
                  (input, h0), 
                  "test_torch_gru.onnx",
                  export_params=True,
                  opset_version=15,
                  do_constant_folding=True,
                  input_names=['input', 'h0'], 
                  output_names=['output', 'hn']
                )
```

### LSTM

```
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(257, 128, 1)
  
    def forward(self, input, h0, c0):
        output, (hn, cn) = self.rnn(input, (h0, c0))
  
        return [output, hn, cn]
  
model = LSTM()
h0 = torch.randn(1, 1, 128)
c0 = torch.randn(1, 1, 128)
input = torch.randn(1, 1, 257)
output, hn, cn = model(input, h0, c0)

torch.onnx.export(model,
                  (input, h0, c0), 
                  "test_torch_lstm.onnx",
                  export_params=True,
                  opset_version=15,
                  do_constant_folding=True,
                  input_names=['input', 'h0', 'c0'], 
                  output_names=['output', 'hn', 'cn']
                )
```

## tensorflow2onnx

### GRU, init_h in Initializer
```
import os

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import concatenate
from keras import regularizers
from keras.constraints import min_max_norm
from keras.constraints import Constraint
from keras import backend as K
import numpy as np


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''

    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


reg = 0.000001
constraint = WeightClip(0.499)

# initial_state = Input(shape=(24), name='h0')
main_input = Input(shape=(None, 38), name='input')
tmp = Dense(24, activation='tanh', name='output_v',
            kernel_constraint=constraint, bias_constraint=constraint)(main_input)
vad_gru, vad_gru2 = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, return_state=True, name='vad_gru', kernel_regularizer=regularizers.l2(
    reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
vad_output = Dense(1, activation='sigmoid', name='output',
                   kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)

model = Model(inputs=[main_input], outputs=[vad_output, vad_gru2])
tf.keras.models.save_model(model, "work_dir/gru")
os.system("python -m tf2onnx.convert --saved-model work_dir/gru --inputs input:0[1,1,38] --opset 13 --output test_gru.onnx")
os.system("onnxsim test_gru.onnx test_gru_sim.onnx")
```

### GRU

```
import os

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import concatenate
from keras import regularizers
from keras.constraints import min_max_norm
from keras.constraints import Constraint
from keras import backend as K
import numpy as np


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''

    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


reg = 0.000001
constraint = WeightClip(0.499)

initial_state_h = Input(shape=(24), name='h0')
initial_state = [initial_state_h]
main_input = Input(shape=(None, 38), name='input')
tmp = Dense(24, activation='tanh', name='output_v',
            kernel_constraint=constraint, bias_constraint=constraint)(main_input)
vad_gru, vad_gru2 = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, return_state=True, name='vad_gru', kernel_regularizer=regularizers.l2(
    reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp, initial_state)
vad_output = Dense(1, activation='sigmoid', name='output',
                   kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)

model = Model(inputs=[main_input, initial_state_h], outputs=[vad_output, vad_gru2])
tf.keras.models.save_model(model, "work_dir/gru")
os.system("python -m tf2onnx.convert --saved-model work_dir/gru --inputs input:0[1,1,38],h0:0[1,24] --opset 13 --output test_gru.onnx")
os.system("onnxsim test_gru.onnx test_gru_sim.onnx")
```

### LSTM

```
import os

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import concatenate
from keras import regularizers
from keras.constraints import min_max_norm
from keras.constraints import Constraint
from keras import backend as K
import numpy as np


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''

    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


reg = 0.000001
constraint = WeightClip(0.499)

initial_state_h = Input(shape=(24), name='h0')
initial_state_c = Input(shape=(24), name='c0')
initial_state = [initial_state_h, initial_state_c]
main_input = Input(shape=(None, 38), name='input')
tmp = Dense(24, activation='tanh', name='output_v',
            kernel_constraint=constraint, bias_constraint=constraint)(main_input)
vad_lstm, vad_lstm2, vad_lstm3 = LSTM(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, return_state=True, name='vad_lstm', kernel_regularizer=regularizers.l2(
    reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp, initial_state)
vad_output = Dense(1, activation='sigmoid', name='output',
                   kernel_constraint=constraint, bias_constraint=constraint)(vad_lstm)

model = Model(inputs=[main_input, initial_state_h, initial_state_c], outputs=[vad_output, vad_lstm2, vad_lstm3])
tf.keras.models.save_model(model, "work_dir/lstm")
os.system("python -m tf2onnx.convert --saved-model work_dir/lstm --inputs input:0[1,1,38],h0:0[1,24],c0:0[1,24] --opset 13 --output test_lstm.onnx")
os.system("onnxsim test_lstm.onnx test_lstm_sim.onnx")
```

## Markdown2pdf

```
img2pdf export_tests/ExportOnnx.jpeg -o export_tests/ExportOnnx.pdf
pandoc export_tests/ExportOnnx.md -o export_tests/ExportOnnx.docx
```
