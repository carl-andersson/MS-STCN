import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import model.torch_ext as torch_ext





class Wavenet(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, downscale=1, upscale=1, in_channels_cond=0):
        super().__init__()
        assert(downscale == 1 or upscale == 1)
        in_channels += in_channels_cond
        seq = []
        dilations = dilation

        if len(dilations) > 1:
            seq.append(WavenetBlock(in_channels, out_channels, kernel_size, dilations[0], downscale=downscale))
            in_channels = out_channels

            for dilation in dilations[1:-1]:
                seq.append(WavenetBlock(in_channels, out_channels, kernel_size, dilation))
                in_channels = out_channels

            seq.append(WavenetBlock(in_channels, out_channels, kernel_size, dilations[-1], upscale=upscale))
        else:
            seq.append(WavenetBlock(in_channels, out_channels, kernel_size, dilations[0], downscale=downscale, upscale=upscale))

        self.layers = nn.Sequential(*seq)

    def forward(self, x, *additional_input):
        x = torch_ext.cat((x, *additional_input), 1)
        return self.layers(x)

class ConditionedWavenet(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, downscale=1, upscale=1, in_channels_cond=0):
        super().__init__()
        assert (downscale == 1 or upscale == 1)

        seq = []
        dilations = dilation

        if len(dilations) > 1:
            seq.append(ConditionedWavenetBlock(in_channels, out_channels, kernel_size, dilations[0],
                                               downscale=downscale, in_channels_cond=in_channels_cond))
            in_channels = out_channels

            for dilation in dilations[1:-1]:
                seq.append(ConditionedWavenetBlock(in_channels, out_channels, kernel_size, dilation,
                                                   in_channels_cond=in_channels_cond))
                in_channels = out_channels

            seq.append(ConditionedWavenetBlock(in_channels, out_channels, kernel_size, dilations[-1],
                                               upscale=upscale, in_channels_cond=in_channels_cond))
        else:
            seq.append(ConditionedWavenetBlock(in_channels, out_channels, kernel_size, dilations[0],
                                               downscale=downscale, upscale=upscale, in_channels_cond=in_channels_cond))

        self.layers = nn.Sequential(*seq)

    def forward(self, x, *cond):
        out = x
        for layer in self.layers:
            out = layer(out, *cond)
        return out


class DownscaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, downscale=1):
        super().__init__()
        self.downscale = downscale
        self.conv = SkipConnection1D(in_channels, out_channels)

        self.register_buffer("downscale_weight", torch.tensor(np.ones((out_channels, 1, downscale))/np.sqrt(downscale),
                                                              dtype=torch.float32))

    def forward(self, x):
        padding = (self.downscale - x.size()[-1] % self.downscale) % self.downscale
        if not np.all(padding == 0):
            padding = [0, padding]
            x = F.pad(x, padding)
        out = self.conv(x)
        out = F.conv1d(out, self.downscale_weight, stride=self.downscale, groups=self.downscale_weight.size(0))
        return out

    @property
    def out_channels(self):
        return self.conv.out_channels

class UpscaleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale=1):
        super().__init__()
        self.upscale = upscale
        self.conv = SkipConnection1D(in_channels, out_channels)
        self.register_buffer("upscale_weight",
                             torch.tensor(np.ones((out_channels, 1, upscale)),
                                          dtype=torch.float32))
    def forward(self, x):
        x = self.conv(x)
        out = F.conv_transpose1d(x, self.upscale_weight, stride=self.upscale, groups=self.upscale_weight.size(0))
        return out

    @property
    def out_channels(self):
        return self.conv.out_channels

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, downscale=1, upscale=1, bias=True):
        super().__init__()
        assert(downscale == 1 or upscale == 1)
        self._kernel_size = kernel_size
        self._dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)

        if upscale > 1:
            self.resample = UpscaleConv1D(out_channels, out_channels, upscale=upscale)
        elif downscale > 1:
            self.resample = DownscaleConv1D(out_channels, out_channels, downscale=downscale)
        else:
            self.resample = nn.Sequential()

        self.receptive_field = (self._kernel_size-1)*self._dilation + 1

    def forward(self, x):
        padding = (self._kernel_size-1)*self._dilation
        if not np.all(padding == 0):
            padding = [padding, 0]
            x = F.pad(x, padding)
        x = self.conv(x)
        return self.resample(x)

    @property
    def out_channels(self):
        return self.conv.out_channels


class GlobalAverageLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.skip = SkipConnection1D(in_channels, out_channels)

    def forward(self, x):
        out = self.skip(x)
        out = out.mean(-1)
        return out




class CausalConvMLP1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, reduction=0.5, activation_fn='elu',
                 kernel_size=1, dilation=1, final_activation=False):
        super().__init__()
        self.out_channels = out_channels
        activation_fn = torch_ext.get_activation_fn(activation_fn)

        layer_sizes = torch_ext.get_MLP_sizes(in_channels, out_channels, n_layers, reduction)

        layers = []
        c_in = in_channels
        for size in layer_sizes:
            layers += [CausalConv1d(c_in, size, kernel_size=kernel_size, dilation=dilation), activation_fn]
            c_in = size

        if final_activation:
            self.layers = nn.Sequential(*layers,
                                        CausalConv1d(c_in, out_channels, kernel_size=kernel_size, dilation=dilation),
                                        activation_fn)
        else:
            self.layers = nn.Sequential(*layers,
                                        CausalConv1d(c_in, out_channels, kernel_size=kernel_size, dilation=dilation))

    def forward(self, x):
        return self.layers(x)


class SkipConnection1D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale=1, downscale=1):
        super().__init__()
        assert(upscale == 1 or downscale == 1)
        self.true_skip = False
        if upscale > 1:
            self.skip = UpscaleConv1D(in_channels, out_channels, upscale=upscale)
        elif downscale > 1:
            self.skip = DownscaleConv1D(in_channels, out_channels, downscale=downscale)
        elif out_channels != in_channels:
            self.skip = CausalConv1d(in_channels, out_channels)
        else:
            self.true_skip = True
            self.skip = nn.Sequential()

    def forward(self, x):
        return self.skip(x)


class ConditionedWavenetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, downscale=1, upscale=1, in_channels_cond=0):
        super().__init__()
        self.out_channels = out_channels

        modules = [CausalConv1d(in_channels + in_channels_cond, 2 * out_channels, kernel_size=kernel_size, dilation=dilation,
                                upscale=upscale, downscale=downscale),
                   torch_ext.AttentionFunction(),
                   CausalConv1d(out_channels, out_channels, bias=False)]

        self.skip = SkipConnection1D(in_channels, out_channels, upscale=upscale, downscale=downscale)

        self.layers = nn.Sequential(*modules)

    def forward(self, x, *cond):
        return self.layers(torch_ext.cat((x, *cond), 1)) + self.skip(x)

class WavenetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, downscale=1, upscale=1, in_channels_cond=0):
        super().__init__()
        in_channels += in_channels_cond
        self.out_channels = out_channels

        modules = [CausalConv1d(in_channels, 2*out_channels, kernel_size=kernel_size, dilation=dilation,
                                upscale=upscale, downscale=downscale),
                   torch_ext.AttentionFunction(),
                   CausalConv1d(out_channels, out_channels, bias=False)]

        self.skip = SkipConnection1D(in_channels, out_channels, upscale=upscale, downscale=downscale)

        self.layers = nn.Sequential(*modules)

    def forward(self, x, *additional_input):
        x = torch_ext.cat((x, *additional_input), 1)
        return self.layers(x) + self.skip(x)



if __name__ == "__main__":
    x = torch.tensor([[[0, 1, 2, 3, 4, 5], [100, 200, 300, 400, 500, 600]]], dtype=torch.float32)

    conv = UpscaleConv1D(2, 2, 4)

    print(x)

    print(conv(x))
