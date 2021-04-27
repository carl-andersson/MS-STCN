# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from scipy.sparse import coo_matrix
from pathlib import Path
import pickle
import os
import urllib
import urllib.request
import numpy as np
from data.base import SeqDataset, Statistics

class PianorollLoader:

    def __init__(self, data_path, dataset, automatic_download):
        if not dataset.endswith(".pkl"):
            dataset += ".pkl"

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        self.file_path = str(Path(data_path).joinpath(dataset))
        if automatic_download:
            self._download_if_needed()


    def _download_if_needed(self):
        if not os.path.exists(self.file_path):
            source_url = "http://www-etud.iro.umontreal.ca/~boulanni/"

            filename_no_ext = Path(self.file_path).stem

            if filename_no_ext == "jsb":
                source_path = source_url +"JSB%20Chorales.pickle"
            elif filename_no_ext == "musedata":
                source_path = source_url + "MuseData.pickle"
            elif filename_no_ext == "nottingham":
                source_path = source_url + "Nottingham.pickle"
            elif filename_no_ext == "piano-midi.de":
                source_path = source_url + "Piano-midi.de.pickle"
            else:
                raise Exception("Unkonwn dataset: "+filename_no_ext)


            filepath, _ = urllib.request.urlretrieve(
                source_path, self.file_path)
            file = os.stat(filepath)
            size = file.st_size
            print('Successfully downloaded', self.file_path, size, 'bytes.')

        else:
            print("Data already downloaded!")
    
    def _sparse_pianoroll_to_dense(self,pianoroll, min_note, num_notes):
        """Converts a sparse pianoroll to a dense numpy array.
        Given a sparse pianoroll, converts it to a dense numpy array of shape
        [num_timesteps, num_notes] where entry i,j is 1.0 if note j is active on
        timestep i and 0.0 otherwise.
        Args:
          pianoroll: A sparse pianoroll object, a list of tuples where the i'th tuple
            contains the indices of the notes active at timestep i.
          min_note: The minimum note in the pianoroll, subtracted from all notes so
            that the minimum note becomes 0.
          num_notes: The number of possible different note indices, determines the
            second dimension of the resulting dense array.
        Returns:
          dense_pianoroll: A [num_timesteps, num_notes] numpy array of floats.
          num_timesteps: A python int, the number of timesteps in the pianoroll.
        """
        num_timesteps = len(pianoroll)
        inds = []
        for time, chord in enumerate(pianoroll):
            # Re-index the notes to start from min_note.
            inds.extend((time, note-min_note) for note in chord)
        shape = [num_timesteps, num_notes]
        values = [1.] * len(inds)
        sparse_pianoroll = coo_matrix(
            (values, ([x[0] for x in inds], [x[1] for x in inds])),
            shape=shape)
        return np.array(sparse_pianoroll.toarray())

    @staticmethod
    def calcStatistics(data):
        cat_data = np.concatenate(data, 0)

        mean = np.mean(cat_data, axis=(0, 1), keepdims=True)
        std = np.std(cat_data, axis=(0, 1), ddof=1, keepdims=True)

        # Change from TxC to CxT
        mean = np.transpose(mean)
        std = np.transpose(std)

        return Statistics(mean, std)

    def loadData(self, seq_len_train, seq_len_valid, seq_len_test, random_crop, min_padding_ratio):
        max_note = 108
        min_note = 21

        num_notes = max_note - min_note + 1
        with open(self.file_path, "rb") as f:
            raw_data = pickle.load(f)

        train_pianorolls = raw_data["train"]
        valid_pianorolls = raw_data["valid"]
        test_pianorolls = raw_data["test"]

        train_data = np.array([self._sparse_pianoroll_to_dense(x, min_note, num_notes) for x in train_pianorolls])
        valid_data = np.array([self._sparse_pianoroll_to_dense(x, min_note, num_notes) for x in valid_pianorolls])
        test_data  = np.array([self._sparse_pianoroll_to_dense(x, min_note, num_notes) for x in test_pianorolls])

        stats = self.calcStatistics(train_data)


        trainset = PianoSeqDataset(train_data, seq_len_train, random_crop=random_crop, min_padding_ratio=min_padding_ratio)


        validset = PianoSeqDataset(valid_data, seq_len_valid)
        train_evalset = PianoSeqDataset(train_data[:len(validset)], seq_len_valid)

        testset = PianoSeqDataset(test_data, seq_len_test)

        return trainset, train_evalset, validset, testset, stats


class PianoSeqDataset(SeqDataset):

    def __init__(self, xs, seq_len, random_crop=False, min_padding_ratio=0.9):
        super().__init__(xs, seq_len, min_padding_ratio=min_padding_ratio)
        self.random_crop = random_crop

    def __getitem__(self, idx):
        if self.random_crop:
            return self._crop(self.x[idx]), 0
        else:
            return self.x[idx], 0

    @staticmethod
    def _crop(x):

        crop_size = ((5, 5), (3, 3))
        x_pad = np.pad(x, crop_size, 'constant')

        nh = np.random.randint(0, crop_size[0][0]+crop_size[0][1]) if crop_size[0][0]+crop_size[0][1] > 0 else 0
        nw = np.random.randint(0, crop_size[1][0]+crop_size[1][1]) if crop_size[1][0]+crop_size[1][1] > 0 else 0
        new_x = x_pad[nh:nh + x.shape[0],
                    nw:nw + x.shape[1]]
        return new_x



def create_pianorolls_dataset(dataset_basepath, data_path, dataset, automatic_download, **params):
    path = str(Path(dataset_basepath).joinpath(Path(data_path)).expanduser())
    return PianorollLoader(data_path=path, dataset=dataset, automatic_download=automatic_download).loadData(**params)


if __name__ == '__main__':
    max_note = 108
    min_note = 21

    num_notes = max_note - min_note + 1
    num_notes = max_note - min_note + 1
    file_path = str(Path.home().joinpath('datasets/pianorolls/').joinpath("jsb.pkl"))
    with open(file_path, "rb") as f:
        raw_data = pickle.load(f)

    train_pianorolls = raw_data["train"]
    valid_pianorolls = raw_data["valid"]
    test_pianorolls = raw_data["test"]

    pl = PianorollLoader("jsb")
    train_data = np.array([pl._sparse_pianoroll_to_dense(x, min_note, num_notes) for x in train_pianorolls])

    import matplotlib.pyplot as plt
    plt.hist([len(d) for d in train_data])
    plt.show()



