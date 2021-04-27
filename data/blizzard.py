import numpy as np
from pathlib import Path
import os
from glob import glob
from scipy.io import wavfile
import subprocess
import tables
from torch.utils.data import Dataset
from data.base import SeqDataLoader, Statistics



def get_and_create_h5(path_to_blizzard, small, sample_len):
    if small:
        path_to_hdf5 = Path(path_to_blizzard).joinpath("data_small.h5")
    else:
        path_to_hdf5 = Path(path_to_blizzard).joinpath("data.h5")

    if not path_to_hdf5.exists():

        path_unsegmented = path_to_blizzard.joinpath("train/unsegmented")
        if not path_unsegmented.exists():
            raise Exception("Missing train/unsegmented in path" + str(path_unsegmented))

        # approx 1147 mp3s
        rng = np.random.RandomState(1)
        all_mp3s = [Path(y) for x in os.walk(path_unsegmented) for y in glob(os.path.join(x[0], '*.mp3'))]
        rng.shuffle(all_mp3s)

        test_split = int(0.9*len(all_mp3s))


        test_mp3s = all_mp3s[test_split:]
        train_mp3s = all_mp3s[:test_split]
        try:
            compression_filter = tables.Filters(complevel=5, complib='blosc')
            with tables.open_file(path_to_hdf5.as_posix(), mode='w') as hdf5_file:
                def create_dataset(name, mp3s):
                    data = hdf5_file.create_earray(hdf5_file.root, name,
                                                   tables.Int16Atom(),
                                                   shape=(0, sample_len),
                                                   filters=compression_filter)
                    path_wav = Path(path_to_blizzard).joinpath("tmp/vaw")
                    path_wav.mkdir(parents=True, exist_ok=True)
                    mean, var, N = 0,0,0;
                    for i, mp3_file in enumerate(mp3s):
                        print("({1}/{2})  {0}".format(mp3_file, i + 1, len(mp3s)))
                        wav_file = path_wav.joinpath(mp3_file.name + ".wav")
                        proc = subprocess.run(["ffmpeg", "-i", mp3_file.as_posix(), "-acodec", "pcm_s16le", "-ac", "1",
                                               "-ar", "16000", wav_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        proc.check_returncode()
                        _, audio_data = wavfile.read(wav_file)
                        if audio_data.ndim > 1:
                            audio_data = audio_data[:, 0]
                        # Cut off trailing and leading zeros due to mp3 decoding
                        zero_idx = np.where(audio_data != 0)[0]
                        audio_data = audio_data[zero_idx[0]:zero_idx[-1] + 1]
                        audio_len = len(audio_data)
                        nsamples = audio_len // sample_len
                        print("Adding {0} samples...".format(nsamples))
                        # Drop last not full sample
                        samples = np.reshape(audio_data[:nsamples * sample_len], (nsamples, sample_len))
                        data.append(samples)
                        os.remove(wav_file)

                        samples_float = samples / 2**15
                        mean += np.sum(samples_float)
                        var += np.sum(samples_float*samples_float)
                        N += samples_float.size
                    return mean / N, (var-mean*mean/N)/N
                print("Training data...")
                mean, var = create_dataset('train', train_mp3s)
                print("Test data...")
                create_dataset('test', test_mp3s)
                print("Done")
                hdf5_file.root.train.attrs["X_mean"] = mean
                hdf5_file.root.train.attrs["X_var"] = var
        except (KeyboardInterrupt, IOError) as e:
            os.remove(path_to_hdf5)
            raise e

    return tables.open_file(path_to_hdf5, mode='r')


def get_datasets(dataset_basepath, data_path, small, frame_size, training_validation_split):
    fullpath = Path(dataset_basepath).joinpath(Path(data_path)).expanduser()
    h5file = get_and_create_h5(fullpath, small, sample_len=8000)
    mean, std = h5file.root.train.attrs["X_mean"], np.sqrt(h5file.root.train.attrs["X_var"])
    len_train = int(len(h5file.root.train)*training_validation_split)

    validset = BlizzardDataset(h5file.root.train, frame_size, mean, std, start_idx=len_train)
    trainset = BlizzardDataset(h5file.root.train, frame_size, mean, std, end_idx=len_train)

    trainset_eval = BlizzardDataset(h5file.root.train, frame_size, mean, std, end_idx=len(validset))
    testset = BlizzardDataset(h5file.root.test, frame_size, mean, std)

    # Already normalized data
    stats = Statistics(0, 1)

    return trainset, trainset_eval, validset, testset, stats




class BlizzardDataset(Dataset):
    def __init__(self, data_table, frame_size, mean=0, std=1, start_idx=0, end_idx=None):
        self.mean = mean
        self.std = std
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.data_table = data_table
        self.frame_size = frame_size
        self.len = (self.end_idx if self.end_idx else len(data_table)) - self.start_idx

    def __getitem__(self, item):
        data = self.data_table[item + self.start_idx] / 2**15
        data = (data-self.mean)/self.std
        return np.reshape(data.astype(np.float32), (self.frame_size, -1)), 0

    def __len__(self):
        return self.len

    @property
    def nc(self):
        return self.frame_size


def play_float_audio(float_audio):
    if np.max(np.abs(float_audio)) > 1:
        float_audio = float_audio / np.max(np.abs(float_audio))
    int_array = np.array(float_audio*2**15).astype(np.int16)
    play_obj = sa.play_buffer(int_array, 1, 2, 16000)
    play_obj.wait_done()


if __name__ == '__main__':
    path_to_blizzard = "~/datasets/blizzard2013"
    trainset, _, _, _, _ = get_datasets(path_to_blizzard, 200, 0.9)

    loader = SeqDataLoader(trainset, batch_size=10, shuffle=False, num_workers=1)

    import simpleaudio as sa

    play_float_audio(trainset[0][0])


    # res = loader.drawBatch(100);
    # print(res.shape)
    # loader.testData()






