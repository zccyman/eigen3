# Operators

| converter operators   | onnx                  | pytorch(torch/torch.nn.functional/torch.nn.Module) |
| --------------------- | --------------------- | -------------------------------------------------- |
| conv/depthwiseconv    | Conv                  | F.conv2 d/nn.Conv2 d                               |
| convtranspose         | ConvTranspose         | F.conv_transpose2 d/nn.ConvTranspose2 d            |
| fc/gemm/matmul        | Gemm/MatMul           | F.linear/nn.Linear                                 |
| lstm                  | LSTM                  | nn.LSTM                                            |
| gru                   | GRU                   | nn.GRU                                             |
| layernormalization    | LayerNormalization    | F.layer_norm/nn.layernorm                          |
| batchnormalization    | BatchNormalization    | F.batch_norm/nn.BatchNorm2 d                       |
| instancenormalization | InstanceNormalization | F.instance_norm/nn.InstanceNorm2 d                 |
| resize                | Resize/Upsample       | F.upsample/F.interpolate/nn.Upsample               |
| relu                  | Relu                  | F.relu/nn.ReLU                                     |
| relu6/relux           | Clip                  | F.relu6/nn.ReLU6                                   |
| sigmoid               | Sigmoid               | F.sigmoid/nn.Sigmoid                               |
| leakyrelu             | LeakyRelu             | F.leaky_relu/nn.LeakyReLU                          |
| tanh                  | Tanh                  | F.tanh/nn.Tanh                                     |
| hardsigmoid           | HardSigmoid           | F.hardsigmoid/nn.Hardsigmoid                       |
| hardswish             | HardSwish             | F.hardswish/nn.Hardswish                           |
| hardtanh              | Clip + Clip           | F.hardtanh/nn.Hardtanh                             |
| hardshrink            |                       | F.hardshrink/nn.Hardshrink                         |
| swish                 | Mul+Sigmoid           | F.silu/nn.SiLU                                     |
| maxpool               | MaxPool               | F.max_pool2 d/nn.MaxPool2 d                        |
| averagepool           | AveragePool           | F.avg_pool2 d/nn.AvgPool2 d                        |
| globalaveragepool     | GlobalAveragePool     | nn.AdaptiveAvgPool2 d(1)                           |
| softmax               | Softmax               | F.softmax/nn.Softmax                               |
| reshape               | Reshape               | torch.reshape                                      |
| flatten               | Flatten               | torch.flatten                                      |
| transpose             | Transpose             | torch.permute                                      |
| unsqueeze             | Unsqueeze             | torch.unsqueeze                                    |
| squeeze               | Squeeze               | torch.squeeze                                      |
| concat                | Concat                | torch.cat                                          |
| split/slice           | Split/Slice           | torch.split                                        |
| add                   | Add                   | +                                                  |
| sub                   | Sub                   | -                                                  |
| mul/cmul/pmul         | Mul                   | *                                                  |

# Markdown2pdf

```
img2pdf export_tests/Operators.jpeg -o export_tests/Operators.pdf
pandoc export_tests/Operators.md -o export_tests/Operators.docx
```
