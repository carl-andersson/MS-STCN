import torch.distributions as tdist
from torch.distributions.utils import broadcast_all
import torch

def move_dim(tensor, dim0, dim1):
    return tensor.unsqueeze(dim1).transpose(dim0, dim1).squeeze(dim0)

class DiagonalMultivariateNormalMixtureND():
    def __init__(self, loc, scale, probs=None, logits=None):
        # loc: NB x n_event x n_mixtures x ND
        # scale: NB x n_event x n_mixtures x ND
        # probs: NB x n_mixtures x ND
        # logits: NB x n_mixtures x ND

        self.before_transform = {"loc": loc, "scale": scale, "probs": probs, "logits": logits}

        # Broadcast and move n_event and n_mixtures last
        if probs is not None:
            probs = probs.unsqueeze(1)
            loc, scale, probs = broadcast_all(loc, scale, probs)
            probs = move_dim(probs, 2, -1).narrow(1, 0, 1).squeeze(1)
        if logits is not None:
            logits = logits.unsqueeze(1)
            loc, scale, logits = broadcast_all(loc, scale, logits)
            logits = move_dim(logits, 2, -1).narrow(1, 0, 1).squeeze(1)

        loc = move_dim(move_dim(loc, 1, -1), 1, -1)
        scale = move_dim(move_dim(scale, 1, -1), 1, -1)

        self._dist = DiagonalMultivariateNormalMixture(loc, scale, probs, logits)

    def sample(self,  sample_shape=torch.Size()):
        # return sample_shape x Batchsize x n_event x  ND
        sample = self._dist.sample(sample_shape)
        sample = move_dim(sample, -1, len(sample_shape) + 1)
        return sample

    def log_prob(self, value):
        # value: NB x n_event x ND
        value = move_dim(value, 1, -1)
        return move_dim(self._dist.log_prob(value), -1, 1)



class DiagonalMultivariateNormalMixture(tdist.Distribution):

    def __init__(self, loc, scale, probs=None, logits=None):
        # loc: ... x n_event x n_mixtures
        # scale: ...  x n_event x n_mixtures
        # probs: ... x n_mixtures
        # logits: ... x n_mixtures

        # Put n_mixtures dimension last
        if probs is not None:
            probs = probs[..., None, :]
            self.loc, self.scale, probs = broadcast_all(loc, scale, probs)
            probs = probs[..., :1, :] # remove n_event dimension
        elif logits is not None:

            logits = logits[..., None, :]
            self.loc, self.scale, logits = broadcast_all(loc, scale, logits)
            logits = logits[..., :1, :]  # remove n_event dimension
        else:
            raise Exception("Either probs or logits must be non-None")

        self.coeff = tdist.Categorical(probs, logits)

        if self.coeff.batch_shape[:-1] != self.loc.size()[:-2]:
            raise Exception("Batchsize and conv_size for normal and mixture coefficients must be equal")
        if self.coeff.param_shape[-1] != self.loc.size()[-1]:
            raise Exception("Normal and mixture coefficients needs to have the same number of mixtures")

        self.n_mixtures = self.coeff.param_shape[-1]

        super().__init__(self.coeff.batch_shape[:-1])

    def sample(self, sample_shape=torch.Size()):
        idx = self.coeff.sample(sample_shape)
        loc_new = self.loc.expand(sample_shape + self.loc.size())
        scale_new = self.scale.expand(sample_shape + self.scale.size())
        # Expand idx in the event dimension and add a dummy dimension for the mixture dimension (needed for gather)
        idx_expanded = idx.expand(loc_new.size()[:-1])[..., None]
        # Gather locations and scales from the indices then remove the dummy dimension from the mixtures
        loc_sampled = torch.gather(loc_new, -1, idx_expanded)[..., 0]
        scale_sampled = torch.gather(scale_new, -1, idx_expanded)[..., 0]

        return tdist.Normal(loc_sampled, scale_sampled).sample()

    def log_prob(self, value):
        log_norm = tdist.Normal(self.loc, self.scale).log_prob(value[..., None])
        log_prob = (log_norm + self.coeff.logits).logsumexp(-1)
        return log_prob



if __name__ == "__main__":

    scale = torch.ones(1, 1, 1, 1)
    loc = torch.zeros(1, 1, 2, 3)
    loc[:, :, 0, :] = -1
    loc[:, :, 1, :] = 1
    loc[:, :, :, 1] *= 2
    loc[:, :, :, 2] *= 3

    probs =  torch.zeros(1, 2, 3)
    probs[:, 0, :] = 0.25
    probs[:, 1, :] = 0.75

    #dist = DiagonalMultivariateNormalMixture(loc=loc, scale=scale, logits=logits)
    dist = DiagonalMultivariateNormalMixtureND(loc=loc, scale=scale, probs=probs)

    sample = dist.sample()
    print(sample.size())
    print(sample)
    print(dist.log_prob(torch.zeros(1, 1, 1, 1)).sum(-1) )







