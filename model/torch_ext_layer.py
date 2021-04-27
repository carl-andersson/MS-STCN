import torch
import model.torch_ext as torch_ext
import torch.nn as nn

class MLPFC(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, reduction=0.5, activation_fn='elu',
                 final_activation=False):
        super().__init__()
        self.out_channels = out_channels
        activation_fn = torch_ext.get_activation_fn(activation_fn)

        layer_sizes = torch_ext.get_MLP_sizes(in_channels, out_channels, n_layers, reduction)

        layers = []
        c_in = in_channels
        for size in layer_sizes:
            layers += [nn.Linear(c_in, size), activation_fn]
            c_in = size

        if final_activation:
            self.layers = nn.Sequential(*layers,
                                        nn.Linear(c_in, out_channels),
                                        activation_fn)
        else:
            self.layers = nn.Sequential(*layers,
                                        nn.Linear(c_in, out_channels))

    def forward(self, x):
        return self.layers(x)

class SkipConnectionFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.true_skip = False
        if out_channels != in_channels:
            self.skip = nn.Linear(in_channels, out_channels)
        else:
            self.true_skip = True
            self.skip = nn.Sequential()

    def forward(self, x):
        return self.skip(x)


class Resnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, in_channels_cond=0):
        super().__init__()
        in_channels += in_channels_cond
        seq = []

        for i in range(n_layers):
            seq.append(ResnetBlock(in_channels, out_channels))
            in_channels = out_channels

        self.layers = nn.Sequential(*seq)

    def forward(self, x, *additional_input):
        x = torch_ext.cat((x, *additional_input), 1)
        return self.layers(x)

class ConditionedRenset(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, in_channels_cond=0):
        super().__init__()
        seq = []

        for i in range(n_layers):
            seq.append(ConditionedResnetBlock(in_channels, out_channels,
                                              in_channels_cond=in_channels_cond))
            in_channels = out_channels

        self.layers = nn.Sequential(*seq)

    def forward(self, x, *cond):
        out = x
        for layer in self.layers:
            out = layer(out, *cond)
        return out



class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_cond=0):
        super().__init__()
        in_channels += in_channels_cond
        self.out_channels = out_channels

        modules = [nn.Linear(in_channels, 2*out_channels),
                   torch_ext.AttentionFunction(),
                   nn.Linear(out_channels, out_channels, bias=False)]

        self.skip = SkipConnectionFC(in_channels, out_channels)

        self.layers = nn.Sequential(*modules)

    def forward(self, x, *additional_input):
        x = torch_ext.cat((x, *additional_input), 1)
        return self.layers(x) + self.skip(x)

class ConditionedResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_cond=0):
        super().__init__()
        self.out_channels = out_channels

        modules = [nn.Linear(in_channels + in_channels_cond, 2 * out_channels),
                   torch_ext.AttentionFunction(),
                   nn.Linear(out_channels, out_channels, bias=False)]

        self.skip = nn.Linear(in_channels, out_channels)

        self.layers = nn.Sequential(*modules)

    def forward(self, x, *cond):
        return self.layers(torch_ext.cat((x, *cond), 1)) + self.skip(x)