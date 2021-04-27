import torch

class Combination:
    def __init__(self, dists, event_sizes):
        self.dists = dists
        self.event_sizes = event_sizes

    def sample(self, sample_shape=torch.Size()):
        out = []
        for dist in self.dists:
            out.append(dist.sample(sample_shape))
        return torch.cat(out, 1)

    def log_prob(self, value):
        values = torch.split(value, self.event_sizes, 1)
        out = []
        for dist, dist_value in zip(self.dists, values):
            out.append(dist.log_prob(dist_value))
        return torch.cat(out, 1)