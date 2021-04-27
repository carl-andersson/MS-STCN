import argparse
import json
import time
import os.path
import train
import logger

import data.loader as loader
from model.ModelState import ModelState
from default_options import default_options

from model.Model import Model

from handle_option_dict import load_checkpoint_options_dict, create_full_options_dict

def get_commandline_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str,
                        help='Directory for logs')

    parser.add_argument('--run_name', type=str,
                        help='The name of this run')

    parser.add_argument('--load_model', type=str, help='Path to a saved model')

    parser.add_argument('--evaluate_model', type=str2bool, const=True, nargs='?', help='Evaluate model')

    parser.add_argument('--cuda', type=str2bool, const=True, nargs='?',
                        help='Specify if the model is to be run on the GPU')

    parser.add_argument('--save_code', type=str2bool, const=True, nargs='?',
                        help='Save code in run folder')

    parser.add_argument('--dataset_basepath', type=str, help='Path to datasets')

    parser.add_argument('--seed', type=int,
                        help='The seed used in pytorch')

    parser.add_argument('--option_file', type=str, default=None,
                        help='File containing Json dict with options (default(%s))')

    parser.add_argument('--option_dict', type=str, default='{}',
                        help='Json with options specified at commandline (default(%s))')

    args = vars(parser.parse_args())

    # Options file
    option_file = args['option_file']

    # Options dict from commandline
    option_dict = json.loads(args['option_dict'])

    commandline_options = {k: v for k, v in args.items() if v is not None and k != "option_file" and k != "option_dict"}

    return commandline_options, option_dict, option_file


def get_run_path(options, ctime):
    logdir = options.get("logdir", None)
    run_name = options.get("run_name", None)

    if not run_name:
        run_name = "train_" + ctime

    if logdir is None:
        logdir = "../log"

    run_path = os.path.abspath(os.path.join(logdir, run_name))
    run_name = os.path.basename(run_path)
    return run_path, run_name


def save_sourcecode(options, run_path, save_code):
    import subprocess
    if save_code:
        subprocess.call(["rsync", "-a", "--exclude-from=../.gitignore", ".", os.path.join(run_path, "code")])
    with open(os.path.join(run_path, "options.txt"), "w+") as f:
        f.write(json.dumps(options, indent=1))
        print(json.dumps(options, indent=1))


def run(options=None, load_model=None, mode_interactive=True):

    if options is None:
        options = {}

    if not mode_interactive:
        # Setup save paths and redirect stdout to file
        ctime = time.strftime("%c")
        run_path, run_name = get_run_path(options, ctime)
        # Create folder
        os.makedirs(run_path, exist_ok=True)

    if load_model is not None:
        #Load saved models options
        options = load_checkpoint_options_dict(os.path.join(os.path.dirname(load_model), "options.txt"), options, default_options)
    else:
        # Use option and fill in with default values
        options = create_full_options_dict(options, default_options=default_options)  # Fill with default values

    if not mode_interactive:
        if load_model is None:
            save_sourcecode(options, run_path, options['save_code'])

        logger.init(False, True, {"project": "MSVAE", "run_name": run_name, 'options': options}, {'logdir': run_path})

        # Set stdout to print to file and console
        logger.set_redirects(run_path)

    if options["dataset"] == "mnist":
        if options["model_options"]["dimensions"] != 2:
            raise Exception("Dataset " + options["dataset"] + " requires 2 dimensions!")
    elif options["dataset"] == "pianoroll":
        if options["model_options"]["dimensions"] != 1:
            raise Exception("Dataset " + options["dataset"] + " requires 1 dimensions!")

    # Load datasets
    loaders, statistics = \
        loader.load_dataset(dataset_basepath=options["dataset_basepath"],
                            dataset=options["dataset"],
                            dataset_options=options["dataset_options"],
                            train_batch_size=options["train_options"]["batch_size"],
                            val_batch_size=options["test_options"]["batch_size"])

    # Define Model
    model = Model(n_channels=loaders["train"].nc, statistics=statistics, **options["model_options"])

    # Setup Modelstate
    modelstate = ModelState(seed=options["seed"],
                            model=model,
                            optimizer=options["optimizer"],
                            optimizer_options=options["optimizer_options"],
                            init_lr=options["train_options"]["init_lr"],
                            init_freebits=options["train_options"]["freebits_options"]["init_threshold"],
                            init_annealing=options["train_options"]["annealing_options"]["init_annealing"])

    if options["cuda"]:
        modelstate.model.cuda()



    if load_model is not None:
        # Restore model
        current_epoch = modelstate.load_model(load_model)
    else:
        current_epoch = 0

    if not mode_interactive:
        print("Training starting at: "+ctime)


        # Run model
        train.run_train(start_epoch=current_epoch,
                        cuda=options["cuda"],
                        ll_normalization=options["ll_normalization"],
                        ind_latentstate=options["ind_latentstate"],
                        modelstate=modelstate,
                        logdir=run_path,
                        loaders=loaders,
                        train_options=options["train_options"],
                        test_options=options["test_options"])
    else:
        return modelstate, loaders, options





if __name__ == "__main__":
    # Get options
    commandline_options, option_dict, option_file = get_commandline_args()
    run_options = create_full_options_dict(commandline_options, option_dict, option_file, default_options=default_options)
    # Run
    run(run_options, load_model=run_options["load_model"], mode_interactive=False)
