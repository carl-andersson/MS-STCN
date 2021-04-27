import torch.distributions as tdist
from torch.distributions.utils import broadcast_all
import torch

def move_dim(tensor, dim0, dim1):
    return tensor.unsqueeze(dim1).transpose(dim0, dim1).squeeze(dim0)

class MultivariateBernoulliMixtureND():
    def __init__(self, mixture_probs=None, mixture_logits=None, coefficient_probs=None, coefficient_logits=None):
        # mixture_probs: NB x n_event x n_mixtures x ND
        # coefficient_probs: NB x n_mixtures x ND

        self.before_transform = {"mixture_probs": mixture_probs, "mixture_logits": mixture_logits,
                                 "coefficient_probs": coefficient_probs, "coefficient_logits": coefficient_logits}

        # Broadcast and move n_event and n_mixtures last
        if mixture_probs is not None and coefficient_probs is not None:
            coefficient_probs = coefficient_probs.unsqueeze(1)
            coefficient_probs, mixture_probs = broadcast_all(coefficient_probs, mixture_probs)
            coefficient_probs = move_dim(coefficient_probs, 2, -1).narrow(1, 0, 1).squeeze(1)
            mixture_probs = move_dim(move_dim(mixture_probs, 1, -1), 1, -1)
        elif mixture_probs is not None and coefficient_logits is not None:
            coefficient_logits = coefficient_logits.unsqueeze(1)
            coefficient_logits, mixture_probs = broadcast_all(coefficient_logits, mixture_probs)
            coefficient_logits = move_dim(coefficient_logits, 2, -1).narrow(1, 0, 1).squeeze(1)
            mixture_probs = move_dim(move_dim(mixture_probs, 1, -1), 1, -1)
        elif mixture_logits is not None and coefficient_probs is not None:
            coefficient_probs = coefficient_probs.unsqueeze(1)
            coefficient_probs, mixture_logits = broadcast_all(coefficient_probs, mixture_logits)
            coefficient_probs = move_dim(coefficient_probs, 2, -1).narrow(1, 0, 1).squeeze(1)
            mixture_logits = move_dim(move_dim(mixture_logits, 1, -1), 1, -1)
        elif mixture_logits is not None and coefficient_logits is not None:
            coefficient_logits = coefficient_logits.unsqueeze(1)
            coefficient_logits, mixture_logits = broadcast_all(coefficient_logits, mixture_logits)
            coefficient_logits = move_dim(coefficient_logits, 2, -1).narrow(1, 0, 1).squeeze(1)
            mixture_logits = move_dim(move_dim(mixture_logits, 1, -1), 1, -1)
        else:
            raise Exception("Either probs or logits must be non-None")

        self._dist = MultivariateBernoulliMixture(mixture_probs=mixture_probs,
                                                  mixture_logits=mixture_logits,
                                                  coefficient_probs=coefficient_probs,
                                                  coefficient_logits=coefficient_logits)

    def sample(self,  sample_shape=torch.Size()):
        # return sample_shape x Batchsize x n_event x ND
        sample = self._dist.sample(sample_shape)
        sample = move_dim(sample, -1, len(sample_shape) + 1)
        return sample

    def log_prob(self, value):
        # value: NB x n_event x ND
        value = move_dim(value, 1, -1)
        return move_dim(self._dist.log_prob(value), -1, 1)



class MultivariateBernoulliMixture(tdist.Distribution):

    def __init__(self, mixture_probs=None, mixture_logits=None, coefficient_probs=None, coefficient_logits=None):
        # mixture_probs: ... x n_event x n_mixtures
        # coefficient_probs: ... x n_mixtures


        if mixture_probs is not None and coefficient_probs is not None:
            coefficient_probs = coefficient_probs.unsqueeze(-2)
            coefficient_probs, mixture_probs = broadcast_all(coefficient_probs, mixture_probs)
            coefficient_probs = coefficient_probs[..., :1, :]
        elif mixture_probs is not None and coefficient_logits is not None:
            coefficient_logits = coefficient_logits.unsqueeze(-2)
            coefficient_logits, mixture_probs = broadcast_all(coefficient_logits, mixture_probs)
            coefficient_logits = coefficient_logits[..., :1, :]
        elif mixture_logits is not None and coefficient_probs is not None:
            coefficient_probs = coefficient_probs.unsqueeze(-2)
            coefficient_probs, mixture_logits = broadcast_all(coefficient_probs, mixture_logits)
            coefficient_probs = coefficient_probs[..., :1, :]
        elif mixture_logits is not None and coefficient_logits is not None:
            coefficient_logits = coefficient_logits.unsqueeze(-2)
            coefficient_logits, mixture_logits = broadcast_all(coefficient_logits, mixture_logits)
            coefficient_logits = coefficient_logits[..., :1, :]
        else:
            raise Exception("Either probs or logits must be non-None")



        self.coeff = tdist.Categorical(coefficient_probs, coefficient_logits)
        self.mixture_probs, self.mixture_logits = mixture_probs, mixture_logits


        self.n_mixtures = self.coeff.param_shape[-1]

        super().__init__(self.coeff.batch_shape[:-1])

    def sample(self, sample_shape=torch.Size()):
        idx = self.coeff.sample(sample_shape)
        if self.mixture_probs is not None:
            mixture_probs_new = self.mixture_probs.expand(sample_shape + self.mixture_probs.size())
            # Expand idx in the event dimension and add a dummy dimension for the mixture dimension (needed for gather)
            idx_expanded = idx.expand(mixture_probs_new.size()[:-1])[..., None]
            # Gather locations and scales from the indices then remove the dummy dimension from the mixtures
            mixture_probs_sampled = torch.gather(mixture_probs_new, -1, idx_expanded)[..., 0]

            return tdist.Bernoulli(probs=mixture_probs_sampled).sample()
        else:
            mixture_logits_new = self.mixture_logits.expand(sample_shape + self.mixture_logits.size())
            # Expand idx in the event dimension and add a dummy dimension for the mixture dimension (needed for gather)
            idx_expanded = idx.expand(mixture_logits_new.size()[:-1])[..., None]
            # Gather locations and scales from the indices then remove the dummy dimension from the mixtures
            mixture_logits_sampled = torch.gather(mixture_logits_new, -1, idx_expanded)[..., 0]
            return tdist.Bernoulli(logits=mixture_logits_sampled).sample()

    def log_prob(self, value):
        log_norm = tdist.Bernoulli(logits=self.mixture_logits, probs=self.mixture_probs).log_prob(value[..., None])
        log_prob = (log_norm + self.coeff.logits).logsumexp(-1)
        return log_prob



if __name__ == "__main__":
    # mixture_probs: NB x n_event x n_mixtures x ND
    mixture_probs = torch.zeros(1, 3, 2, 9)

    mixture_probs[:, 0, 0, :] = 0.1
    mixture_probs[:, 1, 0, :] = 1.0
    mixture_probs[:, 2, 0, :] = 1.0

    mixture_probs[:, 0, 1, :] = 0.5
    mixture_probs[:, 1, 1, :] = 0.5
    mixture_probs[:, 2, 1, :] = 0.0


    probs = torch.zeros(1, 2, 9)
    probs[:, 1, :] = 0.01
    probs[:, 0, :] = 0.99

    dist = MultivariateBernoulliMixtureND(mixture_probs=mixture_probs, coefficient_probs=probs)

    sample = dist.sample()
    print(sample.size())
    print(sample)
    print(dist.log_prob(torch.zeros(1, 3, 9)))







