import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.distributions as tdist
from model.torch_ext import *
from model.Distributions.NormalMixture import DiagonalMultivariateNormalMixtureND
from model.Distributions.Combination import Combination
from model.Distributions.BernoulliMixture import MultivariateBernoulliMixtureND

class DistributionLayer(nn.Module):
    def __init__(self, input_size, output_size, dist_class, dist_options, layer_class, layer_options):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dist = dist_class(output_size=output_size, **dist_options)
        self.layer = layer_class(input_size, self.dist.inputs_required, **layer_options)

    def forward(self, input_data):
        dist_input = self.layer(input_data)
        return self.dist(dist_input)




class CombinationDist(nn.Module):
    def __init__(self, output_size, sub_dists):
        super().__init__()
        self.output_size = output_size
        self.dists = []
        for dist in sub_dists:
            if isinstance(dist, dict):
                dist_type = dist.pop("type")
                if isinstance(dist_type, str):
                    dist_type = get_dist(dist_type)
                self.dists.append(dist_type(**dist))
            else:
                self.dists.append(dist)
        if sum([d.output_size for d in self.dists]) != self.output_size:
            raise Exception("Output sizes must match with total: " +
                            str(sum([d.output_size for d in self.dists])) + " != " + str(self.output_size))

    def forward(self, input_data):
        input_data_per_dist = torch.split(input_data, [d.inputs_required for d in self.dists], 1)
        return Combination([dist(dist_data) for dist, dist_data in zip(self.dists, input_data_per_dist)],
                           [d.output_size for d in self.dists])

    @property
    def inputs_required(self):
        return sum([d.inputs_required for d in self.dists])


class NormalDist(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input_data):
        eta1, eta2 = torch.split(input_data, input_data.size(1)//2, 1)
        eta2 = torch.clamp(F.softplus(eta2), min=1e-3, max=5.0)
        return tdist.Normal(eta1, eta2)

    @property
    def inputs_required(self):
        return self.output_size * 2


class BernoulliDist(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input_data):
        return tdist.Bernoulli(logits=input_data)

    @property
    def inputs_required(self):
        return self.output_size

class NormalMixtureDist(nn.Module):
    def __init__(self, output_size, n_components):
        super().__init__()
        self.n_components = n_components
        self.output_size = output_size


    def forward(self, input_data):
        eta1, eta2 = torch.split(input_data[:, :self.output_size*2*self.n_components, ...], self.output_size*self.n_components, 1)
        eta0 = input_data[:, self.output_size*2*self.n_components:, ...]

        eta2 = torch.clamp(F.softplus(eta2), min=1e-3, max=5.0)
        loc = torch.reshape(eta1, eta1.size()[:1] + (self.output_size, self.n_components) + eta1.size()[2:])
        scale = torch.reshape(eta2, eta2.size()[:1] + (self.output_size, self.n_components) + eta2.size()[2:])
        logits = eta0

        return DiagonalMultivariateNormalMixtureND(loc, scale, logits=logits)

    @property
    def inputs_required(self):
        return self.output_size*2*self.n_components + self.n_components

class BernoulliMixtureDist(nn.Module):
    def __init__(self, output_size, n_components):
        super().__init__()
        self.n_components = n_components
        self.output_size = output_size


    def forward(self, input_data):
        eta1 = input_data[:, :self.output_size*self.n_components, ...]
        eta0 = input_data[:, self.output_size*self.n_components:, ...]

        mixture_logits = torch.reshape(eta1, eta1.size()[:1] + (self.output_size, self.n_components) + eta1.size()[2:])
        logits = eta0

        return MultivariateBernoulliMixtureND(mixture_logits=mixture_logits, coefficient_logits=logits)

    @property
    def inputs_required(self):
        return self.output_size*self.n_components + self.n_components