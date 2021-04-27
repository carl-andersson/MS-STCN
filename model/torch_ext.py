import torch
import torch.nn as nn
import torch.distributions as tdist

import math




def combine(n1: tdist.Normal, n2: tdist.Normal):
    n1var = n1.scale.pow(2)
    n2var = n2.scale.pow(2)
    newloc = (n1.loc * n2var + n2.loc * n1var) / \
             (n1var+n2var)
    newscale = torch.sqrt(n1var * n2var / (n1var + n2var))
    return tdist.Normal(newloc, newscale)

def cat(tensors, dim):
    tensors = tuple(t for t in tensors if t is not None)
    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return tensors[0]
    else:
        return torch.cat(tensors, dim)


def set_weight_norm(module: nn.Module):
    from torch.nn.utils import weight_norm
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        weight_norm(module)

    for c in module.children():
        set_weight_norm(c)


def get_dist(dist_type):
    import model.DistModules as dist_modules
    if dist_type == "bernoulli":
        dist_class = dist_modules.BernoulliDist
    elif dist_type == "bernoulli_mixture":
        dist_class = dist_modules.BernoulliMixtureDist
    elif dist_type == "normal":
        dist_class = dist_modules.NormalDist
    elif dist_type == "normal_mixture":
        dist_class = dist_modules.NormalMixtureDist
    elif dist_type == "combination":
        dist_class = dist_modules.CombinationDist
    else:
        raise Exception("Unknown prediction type" + dist_type)

    return dist_class


def get_activation_fn(activation_fn):
    if activation_fn == 'relu':
        activation_fn = nn.ReLU()
    elif activation_fn == 'elu':
        activation_fn = nn.ELU()
    elif activation_fn == 'sigmoid':
        activation_fn = nn.Sigmoid()
    elif activation_fn == 'none' or not activation_fn:
        activation_fn = nn.Sequential()
    else:
        raise Exception("Activation function not implemented: ", activation_fn)
    return activation_fn

def get_MLP_sizes(in_channels, out_channels, n_layers, reduction):
    diff = in_channels - out_channels
    layer_sizes = []
    for i in range(1, n_layers):
        c_out = math.ceil(min(out_channels, in_channels) + abs(diff) * reduction ** (i if diff > 0 else n_layers - i))
        layer_sizes += [c_out]
    return layer_sizes


class AttentionFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        sig_in, tanh_in = torch.split(input_data, input_data.size(1) // 2, 1)
        return torch.tanh(tanh_in) * torch.sigmoid(sig_in)



def get_class(class_name):
    import torch.nn, model.torch_ext_causal_conv, model.torch_ext_layer
    modules = [torch.nn, model.torch_ext_causal_conv, model.torch_ext_layer]

    for module in modules:
        try:
            return getattr(module, class_name)
        except AttributeError:
            pass
    raise NameError("%s doesn't exist." % class_name)



if __name__ == "__main__":
    #conv = ResNetBlock1D(in_channels=1, out_channels=3, kernel_size=3, stride=2)
    #deconv = ResNetBlock1D(in_channels=3, out_channels=1, kernel_size=3, upscale=2)


    res = get_class("Conv1d")
    print(res)



