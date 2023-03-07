import functools
import math
from bisect import bisect_right

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import source.resnet as res


# from torchvision.models import resnet18


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def muti_cycle(milestones, steps, gamma=0.1):
    milestones = sorted(list(milestones))

    def muti_lr(x):
        n = bisect_right(milestones, x)
        y1 = gamma ** n
        y2 = gamma ** (n + 1)
        already_steps = milestones[min(n - 1, len(milestones) - 1)] if n - 1 >= 0 else 0
        need_steps = milestones[n] if n < len(milestones) else steps
        temp_steps = need_steps - already_steps
        temp_x = x - already_steps
        lr_lambda = ((1 - math.cos(temp_x * math.pi / temp_steps)) / 2) * (y2 - y1) + y1
        return lr_lambda

    return muti_lr


def lr_warmup(optimizer, step, all_steps, lr0, lrf):
    lr = (lrf - lr0) * (step / all_steps) + lr0
    for x in optimizer.param_groups:
        x['lr'] = lr


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class ParamNet(nn.Module):
    def __init__(self, backbone='resnet18', resample_size=128, channels=8, layers=2):
        super(ParamNet, self).__init__()
        assert channels >= 3 and type(channels) is int
        if layers == 1:
            class_num = 12
        elif layers == 2:
            class_num = 7 * channels + 3
        elif layers == 3:
            class_num = 8 * channels + channels * channels + 3
        else:
            raise NotImplementedError("layers must be one of 1,2,3")
        if backbone == 'resnet18':
            self.backbone = res.resnet18(num_classes=class_num)
        elif backbone == 'resnet34':
            self.backbone = res.resnet34(num_classes=class_num)
        elif backbone == 'resnet50':
            self.backbone = res.resnet50(num_classes=class_num)
        elif backbone == 'resnet101':
            self.backbone = res.resnet101(num_classes=class_num)
        else:
            raise NotImplementedError("backbone must be one of resnet 18,34,50,101")
        self.resample_size = resample_size
        self.channels = channels
        self.layers = layers
        self.param = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=True)
        self.data_range = torch.nn.Parameter(torch.tensor(4.5, dtype=torch.float), requires_grad=True)

    def cal_func(self, x):
        x_it, w_it = x
        ch = self.channels
        if self.layers == 1:
            x_out = F.conv2d(torch.unsqueeze(x_it, dim=0),
                             weight=w_it[:9].reshape(3, 3, 1, 1),
                             bias=w_it[9:12].reshape(3))
        elif self.layers == 2:

            x_out = F.conv2d(torch.unsqueeze(x_it, dim=0),
                             weight=w_it[:3 * ch].reshape(ch, 3, 1, 1),
                             bias=w_it[3 * ch:4 * ch].reshape(ch))
            x_out = F.relu(x_out, inplace=True)
            x_out = F.conv2d(x_out,
                             weight=w_it[4 * ch:7 * ch].reshape(3, ch, 1, 1),
                             bias=w_it[7 * ch:7 * ch + 3].reshape(3))
        elif self.layers == 3:
            x_out = F.conv2d(torch.unsqueeze(x_it, dim=0),
                             weight=w_it[:3 * ch].reshape(ch, 3, 1, 1),
                             bias=w_it[3 * ch:4 * ch].reshape(ch))
            x_out = F.relu(x_out, inplace=True)
            x_out = F.conv2d(x_out,
                             weight=w_it[4 * ch:4 * ch + ch * ch].reshape(ch, ch, 1, 1),
                             bias=w_it[4 * ch + ch * ch:5 * ch + ch * ch].reshape(ch))
            x_out = F.relu(x_out, inplace=True)
            x_out = F.conv2d(x_out,
                             weight=w_it[5 * ch + ch * ch:8 * ch + ch * ch].reshape(3, ch, 1, 1),
                             bias=w_it[8 * ch + ch * ch:8 * ch + ch * ch + 3].reshape(3))
        else:
            raise NotImplementedError("layers must be one of 1,2,3")
        return x_out

    def forward(self, x):
        x_resample = F.interpolate(x, size=self.resample_size, mode="bilinear", align_corners=True)
        out_weight = self.backbone(x_resample)
        out_weight = self.data_range * out_weight.tanh()
        out_list = [self.cal_func(data) for data in zip(x, out_weight)]
        taylor_img = torch.cat(out_list, dim=0) + x * self.param
        return taylor_img.tanh()


# class StainNet(nn.Module):
#     def __init__(self, input_nc=3, output_nc=3, n_layer=3, n_channel=32, kernel_size=1):
#         super(StainNet, self).__init__()
#         model_list = [nn.Conv2d(input_nc, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2),
#                       nn.ReLU(True)]
#         for n in range(n_layer - 2):
#             model_list.append(
#                 nn.Conv2d(n_channel, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
#             model_list.append(nn.ReLU(True))
#         model_list.append(nn.Conv2d(n_channel, output_nc, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
#         self.rgb_trans = nn.Sequential(*model_list)
#
#     def forward(self, x):
#         return self.rgb_trans(x)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class GANloss(nn.Module):
    def __init__(self):
        super(GANloss, self).__init__()

    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real).to(pred_real.device))
            loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake).to(pred_fake.device))
            loss = (loss_real + loss_fake) * 0.5
            return loss
        else:
            loss = F.mse_loss(pred_real, torch.ones_like(pred_real).to(pred_real.device))
            return loss


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
