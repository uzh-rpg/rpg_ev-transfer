import torch
import torch.nn as nn
from models.spectral_norm import SpectralNorm


class InterpolationLayer(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(InterpolationLayer, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode

    def forward(self, x):
        if self.scale_factor is not None:
            if self.mode == 'nearest' and self.scale_factor == 2:
                return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1),
                                                                                      2 * x.size(2), 2 * x.size(3))
            else:
                return self.interp(x, scale_factor=self.scale_factor, mode=self.mode)

        else:
            return self.interp(x, size=self.size, mode=self.mode)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 activation='LeakyReLU', norm=None, init_method=None, std=1., sn=False):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if sn:
            self.conv2d = SpectralNorm(self.conv2d)
        if activation is not None:
            if activation == 'LeakyReLU':
                self.activation = getattr(torch.nn, activation, 'LeakyReLU')
                self.activation = self.activation()
            else:
                self.activation = getattr(torch, activation, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=0.01)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None, sn=False):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        if sn:
            self.conv1 = SpectralNorm(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1, bias=bias))
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        if sn:
            self.conv2 = SpectralNorm(
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, bias=bias))
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 10.)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Normalize2D(nn.Module):
    def __init__(self):
        super(Normalize2D, self).__init__()

    def forward(self, x):
        return nn.functional.normalize(x.flatten(start_dim=-2), p=2, dim=1).reshape_as(x)
