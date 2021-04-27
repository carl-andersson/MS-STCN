import torch.nn as nn

from model.MultiscaleVAE import MultiscaleVAE
from model.torch_ext import  set_weight_norm
import torch.distributions as tdist

from model.torch_ext import *
from model.DistModules import DistributionLayer
from model.torch_ext_causal_conv import *
from model.Normalizer import getNormalizerForDistModule


def getPredictLayer(pred_options, input_size, output_size):
    import copy
    pred_options = copy.deepcopy(pred_options)
    pred_type = pred_options.pop("type")
    pred_class = get_dist(pred_type)

    pred = DistributionLayer(input_size, output_size,
                             pred_class, pred_options,
                             CausalConvMLP1d, {'n_layers': 1})
    return pred


class Model(nn.Module):
    def __init__(self, n_channels, data_normalize, statistics, pred_options, dimensions, nn_normalization,
                 vae_options, positional_encoding=None):
        super().__init__()

        if nn_normalization  == "batchnorm":
            weightnorm = False
            batchnorm = True
        elif nn_normalization == "weightnorm":
            weightnorm = True
            batchnorm = False
        elif nn_normalization == "none" or not nn_normalization:
            weightnorm = False
            batchnorm = False
        else:
            raise Exception("Normalization mode not recognized: " + nn_normalization)

        input_size = n_channels
        if positional_encoding is not None:
            #self.positional_encoding = PositionalEncoding(**positional_encoding)
            self.init_layer = SkipConnection1D(input_size, self.positional_encoding.dim)
            input_size = 2*self.positional_encoding.dim
        else:
            self.positional_encoding = None
            self.init_layer = None


        self.vae = MultiscaleVAE(input_size=input_size,
                                 output_size=None,
                                 **vae_options)

        self.pred = getPredictLayer(pred_options, input_size=self.vae.output_size,  output_size=n_channels)

        if data_normalize:
            self.normalizer = getNormalizerForDistModule(self.pred.dist, statistics.mean, statistics.std)
        else:
            self.normalizer = None


        if weightnorm:
            set_weight_norm(self)
        if batchnorm:
            self.vae.apply_batchnorm()

    def forward(self, x, lengths=None, iw_samples=1):
        # returns the prediction and the latent disitributions
        # (#IW_samples*Batch size x Channels x H x W or #IW_samples*Batch size x Channels x W
        if self.normalizer is not None:
            input_data = self.normalizer.normalize(x)
        else:
            input_data = x

        if self.positional_encoding is not None:
            input_data = self.positional_encoding(self.init_layer(input_data))

        if iw_samples > 1:
            input_data = input_data.repeat(iw_samples, *[1]*len(input_data.size()[1:]))
            lengths = lengths.repeat(iw_samples)

        out, ls_arr = self.vae(input_data, lengths=lengths)

        pred = self.pred(out)


        if self.normalizer is not None:
            pred = self.normalizer.unnormalize(pred)

        return pred, ls_arr


    def sample(self, cuda, sample_len, n_samples=1):
        x = torch.zeros((n_samples, self.pred.output_size, 0))
        if cuda:
           x = x.cuda()

        self.vae.reset_sample()

        for i in range(sample_len):
            cond = F.pad(x, [0, 1])
            if self.normalizer is not None:
                cond = self.normalizer.normalize(cond)

            if self.positional_encoding is not None:
                cond = self.positional_encoding(self.init_layer(cond))

            out = self.vae.sample(cond)

            pred = self.pred(out)

            if self.normalizer is not None:
                pred = self.normalizer.unnormalize(pred)

            x = torch.cat([x, pred.sample()], -1)
        return x