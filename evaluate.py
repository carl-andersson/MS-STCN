import run
from model.base import calc_ll_estimate
import numpy as np
import matplotlib.pyplot as plt
import torch
import data.loader as loader
from model.Normalizer import getNormalizerForDistModule


(modelstate, loaders, options) = run.run({'cuda': False, "test_options": {'batch_size': 32}}, load_model="trained_models/ecg_fc2/best_model.pt", mode_interactive=True)
model = modelstate.model

# loaders, statistics = \
#         loader.load_dataset(dataset=options["dataset"],
#                             dataset_options=options["dataset_options"],
#                             train_batch_size=options["train_options"]["batch_size"],
#                             val_batch_size=options["test_options"]["batch_size"])
#
# normalizer = getNormalizerForDistModule(model.pred.dist, statistics.mean, statistics.std)
# model.normalizer = normalizer
#


def plot_lines_example(X, diff=False, yoffset=0):

    X = X.copy().transpose()

    if diff:
        X[:, 1:] = np.cumsum(X[:, 1:], 0)
        X = np.concatenate([np.array([[0, 0, 0]]), X], 0)

    penup_index = np.where(X[:, 0] == 1)[0] + 1

    prev_penup = 0
    for penup in penup_index:
        plt.plot(X[prev_penup:penup, 1], X[prev_penup:penup, 2] + yoffset, color='k', linewidth=1.0)
        prev_penup = penup



def setup_plot(equal=True):
    if equal:
        plt.axis('equal')

def save(save_name="tmp.png"):
    plt.savefig(save_name, dpi=500)

def evluate_speech_text():
    model.eval()
    total_vloss = 0
    n_tot = 0
    for i, (x, y) in enumerate(loaders["test"]):

        vloss = calc_ll_estimate(x, *model(x, iw_samples=1), 1,
                                 1.0, 0, None,
                                 ll_normalization="", ind_latentstate=False).mean()

        batchsize = x.size()[0]
        n_tot += batchsize

        total_vloss += batchsize * vloss.item()

        print(total_vloss / n_tot)

    return total_vloss / n_tot

def weird_sample():
    for i, (x, y) in enumerate(loaders["test"]):
        if i == 30:
            return x

def evaluate_piano(loader, iw_samples=1, musedata=False):
    model.eval()
    total_vloss = 0
    n_tot = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if not i == 30 or not musedata:
                vloss = calc_ll_estimate(x, *model(x, iw_samples=iw_samples), iw_samples,
                                         1.0, 0, None,
                                         ll_normalization="sequence", ind_latentstate=False).mean()

                n_samples = x.size()[0]*x.size()[-1]
                n_tot += n_samples

                total_vloss += n_samples * vloss.item()

                print(i, total_vloss / n_tot)


    return total_vloss / n_tot

def plot(seed=1):
    torch.manual_seed(seed)
    sample = modelstate.model.sample(False, 512, 6).data.numpy()
    plt.figure()
    for i in range(6):
        plot_lines_example(sample[i, ..., :], diff=True, yoffset=20*i)

    save("samples.png")

    plt.figure()
    i = 0
    seqs = [x[0, ..., :512].data.numpy() for (x, y, l) in loaders["train_eval"] if x.shape[-1] > 512]
    np.random.shuffle(seqs)
    for x in seqs:
        i += 1
        plot_lines_example(x, diff=True, yoffset=20 * i)
        if i >= 6:
            break
    save("ground_truth.png")

