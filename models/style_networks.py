"""
Code adapted from: https://github.com/HsinYingLee/DRIT
"""
import torch
import torch.nn as nn
import torchvision.models as models

from models.submodules import InterpolationLayer


class StyleEncoder(nn.Module):
    def __init__(self, input_dim, shared_layers, attribute_channels, use_attributes=False):
        super(StyleEncoder, self).__init__()
        conv_list = []
        self.use_attributes = use_attributes
        self.attribute_channels = attribute_channels

        conv_list += [nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
        conv_list += list(models.resnet18(pretrained=True).children())[1:3]
        conv_list += list(models.resnet18(pretrained=True).children())[4:5]
        conv_list.append(list(models.resnet18(pretrained=True).children())[5][0])

        self.conv_layers = nn.Sequential(*conv_list)
        self.conv_share = shared_layers

        if self.use_attributes:
            # Attribute Layers
            self.attribute_encoder = E_attr_concat(input_dim=input_dim,
                                                   output_nc=attribute_channels, norm_layer=nn.InstanceNorm2d,
                                                   nl_layer=nn.LeakyReLU)

    def forward(self, x, attribute_only=False):
        if self.use_attributes:
            output_mean, output_logvar = self.attribute_encoder(x)
            z_attr = output_mean

            if attribute_only:
                return output_mean, output_logvar, z_attr

        x_conv = self.conv_layers(x)
        x_conv = self.conv_share(x_conv)

        if not self.use_attributes:
            return x_conv, None, None, None

        return x_conv, output_mean, output_logvar, z_attr


class E_attr_concat(nn.Module):
    def __init__(self, input_dim, output_nc=8, norm_layer=None, nl_layer=None):
        super(E_attr_concat, self).__init__()
        ndf = 64
        n_blocks = 4
        max_ndf = 4

        conv_layers_A = [nn.ReflectionPad2d(1)]
        conv_layers_A += [nn.Conv2d(input_dim, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n+1)  # 2**n
            conv_layers_A += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_A += [nl_layer(), nn.AdaptiveAvgPool2d(1)]
        self.fc_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv_layers = nn.Sequential(*conv_layers_A)

    def forward(self, x):

        x_conv = self.conv_layers(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc_A(conv_flat)

        # outputVar = self.fcVar_A(conv_flat)
        outputVar = None

        return output, outputVar


class StyleDecoder(torch.nn.Module):
    def __init__(self, input_c, output_c, attribute_channels, sensor_name):
        super(StyleDecoder, self).__init__()
        self.attribute_channels = attribute_channels
        tch = input_c + self.attribute_channels + 2
        self.index_coords = None

        decoder_list_1 = []

        for i in range(0, 3):
            decoder_list_1 += [INSResBlock(tch, tch)]

        self.decoder_scale_1 = torch.nn.Sequential(*decoder_list_1)
        tch = tch + self.attribute_channels
        self.decoder_scale_2 = nn.Sequential(InterpolationLayer(scale_factor=2, mode='nearest'),
                                             ReLUINSConv2d(tch, tch//2, kernel_size=3, stride=1, padding=1))
        tch = tch // 2
        tch = tch + self.attribute_channels
        self.decoder_scale_3 = nn.Sequential(InterpolationLayer(scale_factor=2, mode='nearest'),
                                             ReLUINSConv2d(tch, tch//2, kernel_size=3, stride=1, padding=1))
        tch = tch // 2
        tch = tch + self.attribute_channels
        self.decoder_scale_4 = nn.Sequential(
                                torch.nn.Conv2d(tch, output_c, kernel_size=1, stride=1, padding=0))

    def forward(self, x, z):
        if self.index_coords is None:
            x_coords = torch.arange(x.size(2), device=x.device, dtype=torch.float)
            y_coords = torch.arange(x.size(3), device=x.device, dtype=torch.float)
            self.index_coords = torch.stack(torch.meshgrid([x_coords,
                                                            y_coords]), dim=0)
            self.index_coords = self.index_coords[None, :, :, :].repeat([x.size(0), 1, 1, 1])

        x = torch.cat([x, self.index_coords], dim=1)

        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z = torch.cat([x, z_img], 1)
        out1 = self.decoder_scale_1(x_and_z)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.decoder_scale_2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.decoder_scale_3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.decoder_scale_4(x_and_z4)

        return out4


class ContentDiscriminator(nn.Module):
    def __init__(self, nr_channels, smaller_input=False):
        super(ContentDiscriminator, self).__init__()
        model = []
        model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=7, stride=2, padding=1, norm='Instance')]
        if smaller_input:
            model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=4, stride=1, padding=1, norm='Instance')]
        else:
            model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=7, stride=1, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(nr_channels, nr_channels, kernel_size=4, stride=1, padding=0)]
        model += [nn.Conv2d(nr_channels, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)

        return out


class CrossDiscriminator(nn.Module):
    def __init__(self, input_dim, n_layer=6, norm='None', sn=True):
        super(CrossDiscriminator, self).__init__()
        ch = 64
        self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] #16
        tch = ch

        for i in range(1, n_layer-1):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] # 8
            tch *= 2

        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)] # 2
        tch *= 2
        if sn:
            model += [torch.nn.utils.spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]  # 1
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1

        return nn.Sequential(*model)

    def cuda(self, gpu):
        self.model.cuda(gpu)

    def forward(self, x_A):
        out_A = self.model(x_A)

        return out_A


####################################################################
# -------------------------- Basic Blocks --------------------------
####################################################################

def conv3x3(in_planes, out_planes):
    return [nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)]
    # return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += conv3x3(inplanes, outplanes)
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


# The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return torch.nn.functional.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape),
                                                  self.bias.expand(normalized_shape))
        else:
            return torch.nn.functional.layer_norm.layer_norm(x, normalized_shape)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += conv3x3(inplanes, inplanes)
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        if sn:
            model += [torch.nn.utils.spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride,
                                                             padding=padding, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        #elif == 'Group'

    def forward(self, x):
        return self.model(x)

class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)]
        # return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]

    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class MisINSResBlock(nn.Module):
    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1))
        # return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))

    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(MisINSResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        model = []
        if dropout > 0:
          model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out += residual
        return out

class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                     output_padding=output_padding, bias=True)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)
