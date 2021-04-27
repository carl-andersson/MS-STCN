import torch
import torch.nn as nn
import numpy as np
from torch.distributions.kl import kl_divergence
import torch.distributions as tdist

def logexpmean(ll, dim):
    return (ll-ll.max(dim, keepdim=True)[0]).exp().mean(dim).log() + ll.max(dim)[0]

class LatentState:

    def __init__(self, posterior, prior, sample, lengths=None):
        self.prior = prior
        self.posterior = posterior
        self.sample = sample
        self.lengths = lengths

    def kl(self):
        res = kl_divergence(self.posterior, self.prior)
        if self.lengths is not None:
            max_len = res.shape[2]
            length_mask = (torch.arange(max_len, device=self.lengths.device)[None, :] < self.lengths[:, None])[:, None, :]
            res = res*length_mask
        return res

    def kl_sample(self):
        res = self.posterior.log_prob(self.sample) - self.prior.log_prob(self.sample)
        if self.lengths is not None:
            max_len = res.shape[2]
            length_mask = (torch.arange(max_len, device=self.lengths.device)[None, :] < self.lengths[:, None])[:, None, :]
            res = res * length_mask
        return res

    def posterior_entropy(self):
        res = self.posterior.enntropy()
        if self.lengths is not None:
            max_len = res.shape[2]
            length_mask = (torch.arange(max_len, device=self.lengths.device)[None, :] < self.lengths[:, None])[:, None, :]
            res = res * length_mask
        return res

    def posterior_entropy_sample(self):
        res = self.posterior.log_prob(self.sample)
        if self.lengths is not None:
            max_len = res.shape[2]
            length_mask = (torch.arange(max_len, device=self.lengths.device)[None, :] < self.lengths[:, None])[:, None, :]
            res = res * length_mask
        return res




def calc_ll_estimate(x, pred, ls_arr, lengths, iw_samples, beta_annealing, freebits, freebits_options, ll_normalization, ind_latentstate):
    # x (B x C x H x W) or (B x C x W)
    # pred distribution with (IB x C x H x W) or (IB x C x W) as sample size
    # ls_arr array of (IB x C x H x W) distributions or (IB x C x W)
    # lengths None or (B)
    if len(x.size()) == 4:
        # 2D data with channel and batch
        if not ind_latentstate:
            summation_dims = (1, 2, 3)
        else:
            summation_dims = (1,)
    elif len(x.size()) == 3:
        # 1D data with channel and batch
        if not ind_latentstate:
            summation_dims = (1, 2)
        else:
            summation_dims = (1,)
    else:
        raise Exception("Unknown data format")

    if iw_samples > 1:
        xs = x.repeat(iw_samples, *[1] * len(x.size()[1:]))
        kl_sum = torch.zeros(xs.size(), device=x.device)
        kl_sample_sum = torch.zeros(xs.size(), device=x.device)
        if lengths is not None:
            max_len = xs.shape[2]
            length_mask = (torch.arange(max_len, device=lengths.device)[None, :] < lengths.repeat(iw_samples)[:, None])[:, None, :]
        else:
            length_mask = None
    else:
        kl_sum = torch.zeros(x.size(), device=x.device)
        kl_sample_sum = torch.zeros(x.size(), device=x.device)

        if lengths is not None:
            max_len = x.shape[2]
            length_mask = (torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None])[:, None, :]
        else:
            length_mask = None

    kl_sum = kl_sum.sum(summation_dims)
    kl_sample_sum = kl_sample_sum.sum(summation_dims)

    if freebits_options and freebits > 0:
        threshold = torch.tensor(freebits, device=x.device, dtype=torch.float32)
        if freebits_options["grouping"] == "channel":
            kl_grouping = summation_dims[:1]
        elif freebits_options["grouping"] == "image" or freebits_options["grouping"] == "sequence":
            kl_grouping = summation_dims[1:]
        elif freebits_options["grouping"] == "full":
            kl_grouping = summation_dims
        elif freebits_options["grouping"] == "none" or not freebits_options["grouping"]:
            kl_grouping = ()
        else:
            raise Exception("Unknown freebits grouping: " + freebits_options["grouping"])

        for ls_elem in ls_arr:
            kl_elem = ls_elem.kl()
            kl_elem_sample = ls_elem.kl_sample()

            if len(kl_elem.size()) == 2:
                kl_elem = kl_elem[..., None]
            if len(kl_elem_sample.size()) == 2:
                kl_elem_sample = kl_elem_sample[..., None]


            if kl_grouping:
                kl_elem = kl_elem.sum(kl_grouping, keepdim=True)
                kl_elem_sample = kl_elem_sample.sum(kl_grouping, keepdim=True)

            if freebits_options["threshold_behaviour"] == "linear":
                kl_elem_sample[threshold > kl_elem] = \
                    (kl_elem.detach() / threshold * kl_elem_sample.clone())[threshold > kl_elem]
                kl_elem[threshold > kl_elem] = \
                    (kl_elem.detach() / threshold * kl_elem.clone())[threshold > kl_elem]
            elif freebits_options["threshold_behaviour"] == "constant":
                kl_elem_sample[threshold > kl_elem] = threshold
                kl_elem = torch.max(kl_elem, threshold)
            else:
                raise Exception("Unknown freebits behaviour: " + freebits_options["threshold_behaviour"])

            kl_sum += kl_elem.sum(summation_dims)
            kl_sample_sum += kl_elem_sample.sum(summation_dims)
    else:
        for ls_elem in ls_arr:
            kl_elem = ls_elem.kl()
            kl_elem_sample = ls_elem.kl_sample()

            kl_sample_sum += kl_elem_sample.sum(summation_dims)
            kl_sum += kl_elem.sum(summation_dims)

    if beta_annealing < 1:
        kl_sample_sum *= beta_annealing
        kl_sum *= beta_annealing

    if iw_samples > 1:
        llx = pred.log_prob(xs)
        if length_mask is not None:
            llx = llx * length_mask

        llx = llx.sum(summation_dims)
        ll = (llx - kl_sample_sum)

        ll = ll.reshape((iw_samples, x.size()[0]))
        ll = logexpmean(ll, 0)

        tot_nll = - ll
    else:
        llx = pred.log_prob(x)
        if length_mask is not None:
            llx = llx * length_mask
        llx = llx.sum(summation_dims)
        tot_nll = -(llx - kl_sum)

    if not ll_normalization:
        tot_nll = tot_nll.mean(0).sum()
    elif ll_normalization == "sequence":
        if lengths is not None:
            tot_nll = (tot_nll/lengths).mean(0).sum()
        else:
            tot_nll = tot_nll.mean(0).sum()/x.size(-1)
    else:
        raise Exception("Unknown ll_normalization: " + ll_normalization)

    return tot_nll
