import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch


class ImageDataLoader(DataLoader):
    # no channels
    @property
    def nc(self):
        return self.dataset.nc

    # image size
    @property
    def size(self):
        return self.dataset.size

    @property
    def nclasses(self):
        return self.dataset.nclasses


class ImageDataset(Dataset):
    """Create dataset from data.

    """
    def __init__(self, xs, ys, nclasses):

        self.x = xs
        self.y = ys
        self._nclasses = nclasses

    @property
    def nc(self):
        return self.x.shape[1]

    @property
    def nclasses(self):
        return self._nclasses

    @property
    def size(self):
        return self.x.shape[2:3]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, ...], self.y[idx, ...]



def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]

    maxLen = max([b[0].shape[-1] for b in batch])
    xs = []
    ys = []
    ls = []
    for b in batch:
        x = b[0]
        y = b[1]
        l = x.shape[-1]
        x = np.pad(x, ((0, 0), (0, maxLen-l)), mode='constant')
        xs.append(x)
        ys.append(y)
        ls.append(l)


    return torch.tensor(xs), torch.tensor(ys), torch.tensor(ls)

class SeqDataLoader(DataLoader):

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)

    # #input channels
    @property
    def nc(self):
        return self.dataset.nc


class SeqDataset(Dataset):
    """Create dataset from data.

    Parameters
    ----------
    x, y: array of ndarrays, shape (total_len, n_channels) or (total_len,)
        All input and output signals. It should be either a 1d array or a 2d array.
    seq_len: int
        Maximum length for a batch on, respectively. If `seq_len` is smaller than the total
        data length, the data will be further divided in batches. If -1 the sequence length is not changed

    """
    def __init__(self, xs, seq_len, min_padding_ratio=1.0):

        self.x = SeqDataset._batchify(xs, seq_len, min_padding_ratio)
        self.ntotbatch = len(self.x)

    @property
    def nc(self):
        return self.x[0].shape[0]

    def __len__(self):
        return self.ntotbatch

    def __getitem__(self, idx):
        return self.x[idx], 0

    @staticmethod
    def _batchify(xs, seq_len, min_padding_ratio=0.75):
        new_xs = []
        for x in xs:
            x = x.transpose().astype(np.float32)
            if seq_len != -1:
                nbatch = x.shape[1] // seq_len
                proposals = np.split(x, np.arange(1, nbatch+1)*seq_len, 1)
                if proposals[-1].shape[1] < min_padding_ratio*seq_len:
                    proposals = proposals[:-1]
                if len(proposals) == 0:
                    print("Warning: Sequence to short to fit any:", seq_len, "Sequence has length:", x.shape[1])
                    continue
                new_xs.extend(proposals)
            else:
                new_xs.append(x)
        return new_xs



class Statistics:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __str__(self):
        return "Mean: "+ str(self.mean) + " Std: " + str(self.std)