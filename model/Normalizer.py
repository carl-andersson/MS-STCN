import torch
import torch.nn as nn
import numpy as np
import torch.distributions as tdist
import model.Distributions.NormalMixture
import model.Distributions.Combination
import model.Distributions.BernoulliMixture

import model.DistModules as dist_modules

class NormalizerNormal(nn.Module):
    _epsilon = 1e-16

    def __init__(self, offset, scale):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32) + self._epsilon)
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float32))

    def forward(self, *args):
        raise Exception("Don't run with forward")

    def normalize(self, data):
        normalized_data = (data-self.offset)/self.scale
        return normalized_data

    def unnormalize(self, dist):

        mu = dist.loc*self.scale + self.offset
        scale = dist.scale*self.scale
        return tdist.Normal(mu, scale)


class NormalizerNormalMixture(NormalizerNormal):
    def unnormalize(self, dist):

        loc = dist.before_transform["loc"]
        scale = dist.before_transform["scale"]
        logits = dist.before_transform["logits"]
        loc = loc * self.scale.unsqueeze(1) + self.offset.unsqueeze(1)
        scale = scale * self.scale.unsqueeze(1)
        return model.Distributions.NormalMixture.DiagonalMultivariateNormalMixtureND(loc=loc, scale=scale,
                                                                                     logits=logits)


class NormalizerBernoulli(nn.Module):
    _epsilon = 1e-3

    def __init__(self, offset):
        super().__init__()
        self.register_buffer('offset', torch.tensor(np.minimum(np.maximum(offset, self._epsilon), 1-self._epsilon),
                                                    dtype=torch.float32))

    def forward(self, *args):
        raise Exception("Don't run with forward")

    def normalize(self, data):
        return data - self.offset

    def unnormalize(self, dist):
        return tdist.Bernoulli(logits=dist.logits + torch.log(self.offset/(1-self.offset)))


class NormalizerBernoulliMixture(NormalizerBernoulli):
    def unnormalize(self, dist):

        mixture_logits = dist.before_transform["mixture_logits"]
        coefficient_logits = dist.before_transform["coefficient_logits"]
        offset = self.offset.unsqueeze(1)
        mixture_logits = mixture_logits + torch.log(offset/(1.0-offset))
        return model.Distributions.BernoulliMixture.MultivariateBernoulliMixtureND(
            mixture_logits=mixture_logits, coefficient_logits=coefficient_logits)



class NormalizerCombination(nn.Module):
    def __init__(self, offset, scale, combination_dists):
        super().__init__()
        self.sizes = [dist.output_size for dist in combination_dists]
        offsets = np.split(offset, np.cumsum(self.sizes))[:-1]
        scales = np.split(scale, np.cumsum(self.sizes))[:-1]
        self.normalizers = nn.ModuleList([getNormalizerForDistModule(dist, dist_offset, dist_scale)
                                          for dist, dist_offset, dist_scale in zip(combination_dists, offsets, scales)])

    def forward(self, *args):
        raise Exception("Don't run with forward")

    def normalize(self, data):
        dist_datas = torch.split(data, self.sizes, 1)
        return torch.cat([normalizer.normalize(dist_data) for normalizer, dist_data in
                          zip(self.normalizers, dist_datas)],
                         1)

    def unnormalize(self, dist):
        return model.Distributions.Combination.Combination(
            [normalizer.unnormalize(dist) for normalizer, dist in zip(self.normalizers, dist.dists)], self.sizes)


def getNormalizerForDistModule(dist, offset, scale):
    if isinstance(dist, dist_modules.NormalDist):
        return NormalizerNormal(offset, scale)
    elif isinstance(dist, dist_modules.BernoulliDist):
        return NormalizerBernoulli(offset)
    elif isinstance(dist, dist_modules.NormalMixtureDist):
        return NormalizerNormalMixture(offset, scale)
    elif isinstance(dist, dist_modules.BernoulliMixtureDist):
        return NormalizerBernoulliMixture(offset)
    elif isinstance(dist, dist_modules.CombinationDist):
        return NormalizerCombination(offset, scale, dist.dists)
    else:
        raise Exception("Normalizer not implemented for distribution: " + dist.__class__.__name__)




def test():
    pred1 = tdist.Normal(torch.zeros(1000, 1), 1)
    pred2 = tdist.Bernoulli(logits=torch.zeros(1000, 1))
    pred = model.Distributions.Combination.Combination([pred1, pred2], [1, 1])

    dist1 = tdist.Normal(30, 10)
    dist2 = tdist.Bernoulli(0.1)

    data1 = dist1.sample((1000, 1))
    data2 = dist2.sample((1000, 1))

    data = torch.cat([data1, data2], 1)

    mean = data.mean(0)
    std = data.std(0)

    print(mean)

    dist_mod = dist_modules.CombinationDist([dist_modules.NormalDist(1), dist_modules.BernoulliDist(1)], 2)
    normalizer = getNormalizerForDistModule(dist_mod, mean.detach().numpy(), std.detach().numpy())

    n_data = normalizer.normalize(data)
    print(n_data.mean(0), n_data.std(0))

    un_pred = normalizer.unnormalize(pred)
    print(un_pred.dists[0].loc[0], un_pred.dists[0].scale[0], un_pred.dists[1].probs[0])

    print(un_pred.log_prob(data).mean(0))



if __name__ == "__main__":
    test()




