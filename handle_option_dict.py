import json
import copy
import os


def recursive_merge(default_dict, new_dict, path=None, allow_new=False):
    deprecated_options = []

    if path is None:
        path = []
    for key in new_dict:
        if key in default_dict:
            if isinstance(default_dict[key], dict) and isinstance(new_dict[key], dict):
                if key in ("dataset_options", "optimizer_options"):
                    recursive_merge(default_dict[key], new_dict[key], path + [str(key)], allow_new=True)
                elif key == "model_options":
                    default_dict[key] = new_dict[key]
                else:
                    recursive_merge(default_dict[key], new_dict[key], path + [str(key)], allow_new=allow_new)
            elif isinstance(default_dict[key], list) and isinstance(new_dict[key], list):
                default_dict[key] = new_dict[key]
            elif isinstance(default_dict[key], (list, dict)) and new_dict[key] is None:
                default_dict[key] = None
            elif isinstance(default_dict[key], (list, dict)) or isinstance(new_dict[key], (dict, list)):
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
            else:
                default_dict[key] = new_dict[key]
        else:
            if allow_new:
                default_dict[key] = new_dict[key]
            elif key in deprecated_options:
                print("Option: " + key + " is deprecated")
            else:
                raise Exception('Illegal option? Default value not found at %s' % '.'.join(path + [str(key)]))
    return default_dict

def merge_list(new_list, default_elem, path=None, allow_new=False):
    ret_list = []
    for i in range(len(new_list)):
        default_elem = {**copy.deepcopy(default_elem), **new_list[i]}
        ret_list.append(default_elem)
    return ret_list

def fix_default_lists(the_dict):
    for key in the_dict:
        if isinstance(the_dict[key], list):
            for elem in the_dict[key]:
                if isinstance(elem, dict):
                    fix_default_lists(elem)
            if len(the_dict[key]) > 0:
                the_dict[key] = merge_list(the_dict[key], the_dict[key][0])
        elif isinstance(the_dict[key], dict):
            fix_default_lists(the_dict[key])

    return  the_dict

def clean_options(options):
    # Remove unused options
    datasets = ['pianoroll', 'blizzard', 'iamondb', 'deepwriting', 'ecg']
    if options["dataset"] not in datasets:
        raise Exception("Unknown dataset: " + options["dataset"])
    dataset_options = options[options["dataset"] + "_options"]
    options["dataset_options"] = recursive_merge(dataset_options, options["dataset_options"])

    optimizers = ['Adam', 'RMSprop']
    if options["optimizer"] not in optimizers:
        raise Exception("Unknown optimizer: " + options["optimizer"])
    optimizer_options = options[options["optimizer"] + "_options"]
    options["optimizer_options"] = recursive_merge(optimizer_options, options["optimizer_options"])

    #clean all
    remove_options = [name + "_options" for name in datasets+optimizers]
    for key in dict(options):
        if key in remove_options:
            del options[key]
    return options


def create_full_options_dict(*option_dicts, default_options=None):
    """
    Merges multiple option dictionaries with the default dictionary and an optional option file specifying options in
    Json format.

    :param option_dicts: Any number of option dictionaries either in the form of a dictionary or a file containg Json dict
    :param default_options: The default dictionary, specifies all valid parameters
    :return: A merged option dictionary giving priority in the order of the input
    """
    if default_options is None:
        raise Exception("A Default options dictionary must be set")

    merged_options = copy.deepcopy(default_options)

    for option_dict in reversed(option_dicts):
        if option_dict is not None:
            if isinstance(option_dict, str):
                with open(option_dict, "r") as file:
                    option_dict = json.loads(file.read())

            merged_options = recursive_merge(merged_options, option_dict)

    # Clear away unused fields and merge model options
    options = clean_options(merged_options)

    #options = fix_default_lists(options)

    return options


def load_checkpoint_options_dict(ckpt_options, runtime_options, default_options):
    ckpt_options = create_full_options_dict(ckpt_options, runtime_options, default_options=default_options)

    runtime_options["dataset"] = ckpt_options["dataset"]
    runtime_options["optimizer"] = ckpt_options["optimizer"]

    runtime_options["optimizer_options"] = ckpt_options["optimizer_options"]
    runtime_options["model_options"] = ckpt_options["model_options"]
    runtime_options["dataset_options"] = ckpt_options["dataset_options"]

    return recursive_merge(ckpt_options, runtime_options)