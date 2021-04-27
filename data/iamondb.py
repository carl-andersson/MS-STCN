import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import fnmatch
import re
from lxml import etree
from pathlib import Path

from data.base import SeqDataset, Statistics

# Modified version of the code from Variational Recurrent Neural Networks (https://github.com/jych/nips2015_vrnn)


def plot_scatter_iamondb_example(X, y=None, equal=True, show=False, save=False,
                                 save_name="tmp.png"):

    rgba_colors = np.zeros((len(X), 4))
    normed = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # for red the first column needs to be one
    rgba_colors[:, 0] = normed[:, 0]

    # for blue last color column needs to be one
    rgba_colors[:, 2] = np.abs(1 - normed[:, 0])

    # the fourth column needs to be alphas
    rgba_colors[:, 3] = np.ones((len(X),)) * .4 + .4 * normed[:, 0]

    if len(X[0]) == 3:
        plt.scatter(X[:, 1], X[:, 2], color=rgba_colors)
    elif len(X[0]) == 2:
        plt.scatter(X[:, 0], X[:, 1], color=rgba_colors)

    if y is not None:
        plt.title(y)

    if equal:
        plt.axis('equal')

    if show:
        if save is True:
            raise ValueError("save cannot be True if show is True!")
        plt.show()
    elif save:
        plt.savefig(save_name)


def plot_lines_iamondb_example(X, y=None, equal=True, show=False, save=False,
                               save_name="tmp.png", diff=False):

    if diff:
        X[:, 1:] = np.cumsum(X[:, 1:], 0)
        X = np.concatenate([np.array([[0, 0, 0]]), X], 0)

    penup_index = np.where(X[:, 0] == 1)[0] + 1

    prev_penup = 0
    for penup in penup_index:
        plt.plot(X[prev_penup:penup, 1], X[prev_penup:penup, 2], color='k')
        prev_penup = penup


    if y is not None:
        plt.title(y)

    if equal:
        plt.axis('equal')

    if show:
        if save is True:
            raise ValueError("save cannot be True if show is True!")
        plt.show()
    elif save:
        plt.savefig(save_name)


def fetch_iamondb(data_path):

    strokes_path = os.path.join(data_path, "lineStrokes")
    ascii_path = os.path.join(data_path, "ascii")
    train_files_path = os.path.join(data_path, "train.txt")
    valid_files_path = os.path.join(data_path, "valid.txt")
    test_files_path = os.path.join(data_path, "test.txt")

    if not os.path.exists(strokes_path) or not os.path.exists(ascii_path):
        raise ValueError("You must download the data from IAMOnDB, and"
                         "unpack in %s" % data_path)

    if not os.path.exists(train_files_path) or not os.path.exists(valid_files_path):
        raise ValueError("Cannot find concatenated train.txt and valid.txt"
                         "files! See the README in %s" % data_path)

    partial_path = data_path
    train_names = [f.strip()
                   for f in open(train_files_path, mode='r').readlines()]
    valid_names = [f.strip()
                   for f in open(valid_files_path, mode='r').readlines()]
    test_names = [f.strip()
                   for f in open(test_files_path, mode='r').readlines()]

    def construct_ascii_path(f):

        primary_dir = f.split("-")[0]

        if f[-1].isalpha():
            sub_dir = f[:-1]
        else:
            sub_dir = f

        file_path = os.path.join(ascii_path, primary_dir, sub_dir, f + ".txt")

        return file_path

    def construct_stroke_paths(f):

        primary_dir = f.split("-")[0]

        if f[-1].isalpha():
            sub_dir = f[:-1]
        else:
            sub_dir = f

        files_path = os.path.join(strokes_path, primary_dir, sub_dir)

        #Dash is crucial to obtain correct match!
        files = fnmatch.filter(os.listdir(files_path), f + "-*.xml")
        files = [os.path.join(files_path, fi) for fi in files]
        files = sorted(files, key=lambda x: int(x.split(os.sep)[-1].split("-")[-1][:-4]))

        return files

    train_ascii_files = [construct_ascii_path(f) for f in train_names]
    valid_ascii_files = [construct_ascii_path(f) for f in valid_names]
    test_ascii_files = [construct_ascii_path(f) for f in test_names]

    train_stroke_files = [construct_stroke_paths(f) for f in train_names]
    valid_stroke_files = [construct_stroke_paths(f) for f in valid_names]
    test_stroke_files = [construct_stroke_paths(f) for f in test_names]

    train_npy_x = os.path.join(partial_path, "train_npy_x.npy")
    train_npy_y = os.path.join(partial_path, "train_npy_y.npy")
    valid_npy_x = os.path.join(partial_path, "valid_npy_x.npy")
    valid_npy_y = os.path.join(partial_path, "valid_npy_y.npy")
    test_npy_x = os.path.join(partial_path, "test_npy_x.npy")
    test_npy_y = os.path.join(partial_path, "test_npy_y.npy")

    train_set = (list(zip(train_stroke_files, train_ascii_files)),
                 train_npy_x, train_npy_y)

    valid_set = (list(zip(valid_stroke_files, valid_ascii_files)),
                 valid_npy_x, valid_npy_y)
    test_set = (list(zip(test_stroke_files, test_ascii_files)),
                 test_npy_x, test_npy_y)

    if not os.path.exists(train_npy_x):
        for se, x_npy_file, y_npy_file in [train_set, valid_set, test_set]:
            x_set = []
            y_set = []

            for n, (strokes_files, ascii_file) in enumerate(se):
                if n % 100 == 0:
                    print("Processing file %i of %i" % (n, len(se)))
                with open(ascii_file) as fp:
                    cleaned = [t.strip() for t in fp.readlines()
                               if t
                               and t.strip()]

                    # Try using CSR
                    idx = [n for
                           n, li in enumerate(cleaned) if li == "CSR:"][0]
                    cleaned_sub = cleaned[idx + 1:]
                    corrected_sub = []

                    for li in cleaned_sub:
                        # Handle edge case with %%%%% meaning new line?
                        if "%" in li:
                            li2 = re.sub('\%\%+', '%', li).split("%")
                            li2 = [l.strip() for l in li2]
                            corrected_sub.extend(li2)
                        else:
                            corrected_sub.append(li)

                n_one_hot = 57
                y = [np.zeros((len(li), n_one_hot), dtype='int16')
                     for li in corrected_sub]

                # A-Z, a-z, space, apostrophe, comma, period
                charset = list(range(65, 90 + 1)) + list(range(97, 122 + 1)) + [32, 39, 44, 46]
                tmap = {k: n + 1 for n, k in enumerate(charset)}

                # 0 for UNK/other
                tmap[0] = 0

                def tokenize_ind(line):

                    t = [ord(c) if ord(c) in charset else 0 for c in line]
                    r = [tmap[i] for i in t]

                    return r

                for n, li in enumerate(corrected_sub):
                    y[n][np.arange(len(li)), tokenize_ind(li)] = 1

                x = []

                for stroke_file in strokes_files:
                    with open(stroke_file) as fp:
                        tree = etree.parse(fp)
                        root = tree.getroot()
                        # Get all the values from the XML
                        # 0th index is stroke ID, will become up/down
                        s = np.array([[i, int(Point.attrib['x']),
                                      int(Point.attrib['y'])]
                                      for StrokeSet in root
                                      for i, Stroke in enumerate(StrokeSet)
                                      for Point in Stroke])

                        # flip y axis
                        s[:, 2] = -s[:, 2]

                        # Get end of stroke points
                        c = s[1:, 0] != s[:-1, 0]
                        ci = np.where(c == True)[0]
                        nci = np.where(c == False)[0]

                        # set pen down
                        s[0, 0] = 0
                        s[nci, 0] = 0

                        # set pen up
                        s[ci, 0] = 1
                        s[-1, 0] = 1
                        x.append(s)

                if len(x) != len(y):
                    print("Dataset error - len(x) !+= len(y)! " + str(len(x)) + "!=" + str(len(y)))
                    print(corrected_sub)
                    raise ValueError("Error in: "+ ascii_file + " and " + stroke_file)

                x_set.extend(x)
                y_set.extend(y)

            pickle.dump(x_set, open(x_npy_file, mode="wb"))
            pickle.dump(y_set, open(y_npy_file, mode="wb"))

    train_x = pickle.load(open(train_npy_x, mode="rb"))
    train_y = pickle.load(open(train_npy_y, mode="rb"))
    valid_x = pickle.load(open(valid_npy_x, mode="rb"))
    valid_y = pickle.load(open(valid_npy_y, mode="rb"))
    test_x = pickle.load(open(test_npy_x, mode="rb"))
    test_y = pickle.load(open(test_npy_y, mode="rb"))

    return (train_x, train_y, valid_x, valid_y, test_x, test_y)


def calcStatistics(data):
    cat_data = np.concatenate(data, 0)
    mean = np.mean(cat_data, 0, keepdims=True)
    std = np.std(cat_data, 0, ddof=1, keepdims=True)

    # Change from TxC to CxT
    mean = np.transpose(mean)
    std = np.transpose(std)

    return Statistics(mean, std)

def preprocess(all_data):
    res = []
    for data in all_data:
        diff_pos = data[1:, 1:] - data[:-1, 1:]
        res.append(np.concatenate([data[1:, :1], diff_pos], 1))
    return res

def normalize_scale(all_data, scale):
    res = []
    for data in all_data:
        res.append(np.concatenate([data[:, :1], data[:, 1:]/scale], 1))
    return res

def addMeasNoise(all_data, measure_noise):
    res = []
    for data in all_data:
        noise = (np.random.uniform(size=data[:, 1:].shape) - 0.5) * measure_noise
        res.append(np.concatenate([data[:, :1], data[:, 1:] + noise], 1))
    return res


def getDatasets(dataset_basepath, data_path, seq_len_train, seq_len_valid, seq_len_test, min_padding_ratio):
    # Testset not implemented yet, neither are outputs
    data_path = Path(dataset_basepath).joinpath(data_path).expanduser()
    train_x, _, valid_x, _, test_x, _= fetch_iamondb(data_path)

    # Set to relative output
    train_x = preprocess(train_x)
    valid_x = preprocess(valid_x)
    test_x = preprocess(test_x)

    # Calculate scale for comparable ll calculations
    X = [x[:, 1:] for x in train_x]
    X_len = np.array([len(x) for x in X]).sum()
    X_mean = np.array([x.sum() for x in X]).sum() / X_len
    X_sqr = np.array([(x ** 2).sum() for x in X]).sum() / X_len
    X_std = np.sqrt(X_sqr - X_mean ** 2)
    #cat_data = np.concatenate(train_x, 0)
    #scale = np.std(cat_data, ddof=1)

    # Add measurement noise to avoid overfitting. The numbers below are the resolution before normalization
    measure_noise = np.array([1, 1])
    train_x = addMeasNoise(train_x, measure_noise)

    # We do not normalize the mean here since this can be done in the model instead.
    # This still gives the same estimates of the log likelihood
    train_x = normalize_scale(train_x, X_std)
    valid_x = normalize_scale(valid_x, X_std)
    test_x = normalize_scale(test_x, X_std)
    model_stats = calcStatistics(train_x)

    trainset = SeqDataset(train_x, seq_len_train, min_padding_ratio=min_padding_ratio)

    validset = SeqDataset(valid_x, seq_len_valid, min_padding_ratio=min_padding_ratio)
    trainset_eval = SeqDataset(train_x[:len(valid_x)], seq_len_valid, min_padding_ratio=min_padding_ratio)

    testset = SeqDataset(test_x, seq_len_test, min_padding_ratio=min_padding_ratio)

    return trainset, trainset_eval, validset, testset, model_stats



if __name__ == "__main__":
    train_x, train_y, valid_x, valid_y, test_x, test_y = fetch_iamondb("/home/calle/datasets/IAM-OnDB")

    train_x = preprocess(train_x)
    valid_x = preprocess(valid_x)
    stats = calcStatistics(train_x)


    print(train_x[0])
    measure_noise = np.array([1, 1])
    train_x = addMeasNoise(train_x, measure_noise)
    print(train_x[0])
    quit()
    train_x = normalize_scale(train_x, stats)
    print(stats)

    lens = [len(x) for x in train_x]

    print(min(lens), max(lens))
    plt.hist(lens,20)
    plt.show()

    plot_lines_iamondb_example(np.array(train_x[500]), show=True, diff=True)
