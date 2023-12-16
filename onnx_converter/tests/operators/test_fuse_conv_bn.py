import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

myconv = nn.ConvTranspose2d  # nn.ConvTranspose2d | nn.Conv2d


class FusedConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(FusedConvBatchNorm, self).__init__()
        self.conv = myconv(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels, track_running_stats=True)

        self.conv.weight.data.normal_(0, 0.02)  # 初始化卷积权重
        self.conv.bias.data.normal_(0, 0.02)  # 初始化卷积偏置
        self.batch_norm.weight.data.normal_(1.0, 0.02)  # 初始化BN权重
        self.batch_norm.bias.data.normal_(1.0, 0.02)  # 初始化BN偏置
        self.batch_norm.running_mean.data.normal_(1.0, 0.02)
        self.batch_norm.running_var.data.normal_(1.0, 0.02)

        self.conv_fused = myconv(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.conv_fused.weight.data = self.conv.weight.data.clone()
        self.conv_fused.bias.data = self.conv.bias.data.clone()

    def fuse_parameters(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias
        bn_weight = self.batch_norm.weight
        bn_bias = self.batch_norm.bias
        bn_running_mean = self.batch_norm.running_mean
        bn_running_var = self.batch_norm.running_var
        bn_eps = self.batch_norm.eps

        fused_gamma = bn_weight / torch.sqrt(bn_running_var + bn_eps)
        if myconv == nn.ConvTranspose2d:
            fused_weight = conv_weight * fused_gamma.reshape(1, -1, 1, 1)
        else:
            fused_weight = conv_weight * fused_gamma.reshape(-1, 1, 1, 1)
        fused_bias = (conv_bias - bn_running_mean) * fused_gamma + bn_bias

        return fused_weight, fused_bias

    def forward(self, x, use_bn_fused=True):
        if use_bn_fused:
            fused_weight, fused_bias = self.fuse_parameters()
            self.conv_fused.weight.data = fused_weight
            self.conv_fused.bias.data = fused_bias
            y = self.conv_fused(x)
        else:
            y = F.batch_norm(
                self.conv(x),
                running_mean=self.batch_norm.running_mean,
                running_var=self.batch_norm.running_var,
                weight=self.batch_norm.weight,
                bias=self.batch_norm.bias,
                training=False,
                eps=self.batch_norm.eps,
            )

        return y


class TestFusedConvBatchNorm(unittest.TestCase):
    def test_fused_convolution(self):
        in_channels = 3
        out_channels = 64
        kernel_size = 4
        stride = 2
        padding = 1

        input = torch.randn(1, in_channels, 32, 32)
        conv_bn = FusedConvBatchNorm(
            in_channels, out_channels, kernel_size, stride, padding
        )
        output = conv_bn(input, use_bn_fused=True)
        expected_output = conv_bn(input, use_bn_fused=False)

        self.assertTrue(torch.allclose(output, expected_output, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
