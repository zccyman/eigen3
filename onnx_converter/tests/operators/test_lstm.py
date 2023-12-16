# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : henson.zhang
# @Company  : SHIQING TECH
# @Time     : 2022/4/18 9:58
# @File     : test_lstm.py
import sys  # NOQA: E402

sys.path.append('./')  # NOQA: E402

import os
from typing import Any, List, Optional, Sequence, Text, Union

import numpy as np
import onnx
import onnxruntime as rt
import torch
import torch.nn as nn
from onnx.onnx_pb import AttributeProto, FunctionProto, NodeProto, TypeProto

_TargetOpType = ""


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.slice_num = 4
        self.weight_ih = nn.Parameter(
            torch.randn(self.slice_num * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(self.slice_num * hidden_size, hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.randn(self.slice_num * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(self.slice_num * hidden_size))

        self.fc1 = nn.Linear(input_size, self.slice_num * hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.slice_num * hidden_size)

    def forward(self, input, states, param=None):
        if param:
            self.fc1.weight = self.weight_ih = param[0]
            self.fc2.weight = self.weight_hh = param[1]
            self.fc1.bias = self.bias_ih = param[2]
            self.fc2.bias = self.bias_hh = param[3]

        hx, cx = states
        gates = self.fc1(input) + self.fc2(hx)
        # gates = (
        #     torch.mm(input, self.weight_ih.t())
        #     + self.bias_ih
        #     + torch.mm(hx, self.weight_hh.t())
        #     + self.bias_hh
        # )
        ingate, forgetgate, cellgate, outgate = torch.chunk(
            gates, self.slice_num, dim=1
        )
        # ingate, forgetgate, cellgate, outgate = torch.split(
        #     gates, [self.hidden_size for _ in range(self.slice_num)], dim=1
        # )

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cell = LSTMCell(
            input_size=self.input_size, hidden_size=self.hidden_size
        )

    def forward(self, inputs, states, params=None):
        time_step = inputs.shape[0]
        hx_outs = []
        cx_outs = []

        for layer_id in range(self.num_layers):
            if params:
                param = params[layer_id]
            else:
                param = None
            output = []
            hy, cy = states[0][layer_id], states[1][layer_id]
            for time in range(time_step):
                input = inputs[time]
                hy, (hy, cy) = self.lstm_cell(input, (hy, cy), param=param)
                output.append(hy.unsqueeze(dim=0))
            hx_outs.append(hy.unsqueeze(dim=0))
            cx_outs.append(cy.unsqueeze(dim=0))
            outputs = torch.concat(output, dim=0)
            inputs = outputs

        hx_outs = torch.concat(hx_outs, dim=0)
        cx_outs = torch.concat(cx_outs, dim=0)

        return outputs, (hx_outs, cx_outs)

class LSTM_TORCH(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(LSTM_TORCH, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cell = nn.LSTM(input_size, hidden_size,
                         num_layers, bidirectional=bidirectional)

        # self.fc1 = nn.Linear(20, 20)
        # self.fc2 = nn.Linear(20, 20)
        # self.fc3 = nn.Linear(20, 20)

    def forward(self, inputs, states):
        outputs, (hx_outs, cx_outs) = self.lstm_cell(inputs, states)

        # outputs = self.fc1(outputs.permute(1, 0, 2))
        # hx_outs = self.fc2(hx_outs)
        # cx_outs = self.fc3(cx_outs)

        return outputs, (hx_outs, cx_outs)

def test_lstm():
    bidirectional = False
    batch = 1 #2
    input_size = 10
    hidden_size = 20
    num_layers = 1 #2
    time_step = 5

    lstm_torch = LSTM_TORCH(input_size, hidden_size, num_layers, bidirectional=bidirectional)
    params = lstm_torch.lstm_cell.all_weights
    lstm = LSTM(input_size, hidden_size, num_layers)
    xi = torch.randn(
        time_step, batch, input_size
    )
    h0 = torch.randn(num_layers, batch, hidden_size)
    c0 = torch.randn(num_layers, batch, hidden_size)
    xo, (hn, cn) = lstm(xi, (h0, c0), params=params)

    xo_t, (hn_t, cn_t) = lstm_torch(xi, (h0, c0))
    print("=> lstm: ", xo.sum().detach().numpy(),
          hn.sum().detach().numpy(), cn.sum().detach().numpy())
    print("=> lstm_torch: ", xo_t.sum().detach().numpy(),
          hn_t.sum().detach().numpy(), cn_t.sum().detach().numpy())
    if not os.path.exists('work_dir'):
        os.makedirs('work_dir')
    torch.onnx.export(lstm_torch,
                      (xi, (h0, c0)),
                      'work_dir/lstm.onnx',
                      export_params=True,
                      opset_version=14,
                      do_constant_folding=True,
                      input_names=['input1', 'input2'],
                      output_names=['output1', 'output2'])

    sess = rt.InferenceSession('work_dir/lstm.onnx')
    xi_name = sess.get_inputs()[0].name
    h0_name = sess.get_inputs()[1].name
    c0_name = sess.get_inputs()[2].name
    xo_name = sess.get_outputs()[0].name
    hn_name = sess.get_outputs()[1].name
    cn_name = sess.get_outputs()[2].name

    pred_onnx = sess.run([xo_name, hn_name, cn_name], {
        xi_name: xi.numpy(),
        h0_name: h0.numpy(),
        c0_name: c0.numpy()
    })
    xo_onnx = pred_onnx[0]
    hn_onnx = pred_onnx[1]
    cn_onnx = pred_onnx[2]
    xo = xo.detach().numpy()
    hn = hn.detach().numpy()
    cn = cn.detach().numpy()
    error_xo = np.sum(np.abs(xo_onnx - xo)) / np.sum(np.abs(xo))
    error_hn = np.sum(np.abs(hn_onnx - hn)) / np.sum(np.abs(hn))
    error_cn = np.sum(np.abs(cn_onnx - cn)) / np.sum(np.abs(cn))
    print('=> error: ', error_xo, error_hn, error_cn)


if __name__ == '__main__':
    test_lstm()
