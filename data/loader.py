from data.base import ImageDataLoader, SeqDataLoader
from data.pianoroll import create_pianorolls_dataset
import data.blizzard as blizzard
import data.iamondb as iamondb
import data.deepwriting as deepwriting


def load_dataset(dataset_basepath, dataset, dataset_options, train_batch_size, val_batch_size):
    if dataset == 'pianoroll':
        dataset_train, dataset_train_eval, dataset_valid, dataset_test, stats = \
            create_pianorolls_dataset(dataset_basepath, **dataset_options)
        test_batch_size = val_batch_size

        if dataset_options["seq_len_train"] == -1:
            train_batch_size = 1

        if dataset_options["seq_len_valid"] == -1:
            val_batch_size = 1

        if dataset_options["seq_len_test"] == -1:
            test_batch_size = 1

        loader_train = SeqDataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=1)
        loader_train_eval = SeqDataLoader(dataset_train_eval, batch_size=val_batch_size, shuffle=False, num_workers=1)
        loader_valid = SeqDataLoader(dataset_valid, batch_size=val_batch_size, shuffle=False, num_workers=1)
        loader_test = SeqDataLoader(dataset_test,  batch_size=test_batch_size, shuffle=False, num_workers=1)

    elif dataset == 'blizzard':
        dataset_train, dataset_train_eval, dataset_valid, dataset_test, stats = \
            blizzard.get_datasets(dataset_basepath, **dataset_options)
        test_batch_size = val_batch_size
        # Does only support single threaded loading
        loader_train = SeqDataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=1)
        loader_train_eval = SeqDataLoader(dataset_train_eval, batch_size=val_batch_size, shuffle=False, num_workers=1)
        loader_valid = SeqDataLoader(dataset_valid, batch_size=val_batch_size, shuffle=False, num_workers=1)
        loader_test = SeqDataLoader(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
    elif dataset == 'iamondb':
        dataset_train, dataset_train_eval, dataset_valid, dataset_test, stats = \
            iamondb.getDatasets(dataset_basepath, **dataset_options)
        test_batch_size = val_batch_size

        if dataset_options["seq_len_train"] == -1:
            train_batch_size = 1

        if dataset_options["seq_len_valid"] == -1:
            val_batch_size = 1

        if dataset_options["seq_len_test"] == -1:
            test_batch_size = 1

        loader_train = SeqDataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, drop_last=True,
                                     num_workers=1)
        loader_train_eval = SeqDataLoader(dataset_train_eval, batch_size=val_batch_size, shuffle=False, num_workers=1)
        loader_valid = SeqDataLoader(dataset_valid, batch_size=val_batch_size, shuffle=False, num_workers=1)
        loader_test = SeqDataLoader(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
    elif dataset == 'deepwriting':
        dataset_train, dataset_train_eval, dataset_valid, dataset_test, stats = \
            deepwriting.getDatasets(dataset_basepath, **dataset_options)
        test_batch_size = val_batch_size

        if dataset_options["seq_len_train"] == -1:
            train_batch_size = 1

        if dataset_options["seq_len_valid"] == -1:
            val_batch_size = 1

        if dataset_options["seq_len_test"] == -1:
            test_batch_size = 1

        loader_train = SeqDataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, drop_last=True,
                                     num_workers=1)
        loader_train_eval = SeqDataLoader(dataset_train_eval, batch_size=val_batch_size, shuffle=False, num_workers=1)
        loader_valid = SeqDataLoader(dataset_valid, batch_size=val_batch_size, shuffle=False, num_workers=1)
        loader_test = SeqDataLoader(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
    else:
        raise Exception("Dataset not implemented: {}".format(dataset))

    return {"train": loader_train, "train_eval":loader_train_eval,  "valid": loader_valid, "test": loader_test}, stats


if __name__ == "__main__":
    load_dataset("mnist", {}, 100, 100)
