default_pianoroll_options = {
    'data_path': "pianorolls",
    'dataset': 'jsb',
    'automatic_download': True,
    'seq_len_train': 32,
    'seq_len_valid': -1,
    'seq_len_test': -1,
    'random_crop': True,
    'min_padding_ratio': 1.0
}

default_blizzard_options = {
    'data_path': "blizzard2013",
    'small': False,
    "frame_size": 2,
    "training_validation_split": 0.95
}

default_iamondb_options = {
    'seq_len_train': 1024,
    'seq_len_valid': -1,
    'seq_len_test': -1,
    'data_path': "IAM-OnDB",
    'min_padding_ratio': 0.05
}

default_deepwriting_options = {
    'seq_len_train': 128,
    'seq_len_valid': -1,
    'seq_len_test': -1,
    'data_path': "deepwriting",
    'min_padding_ratio': 0.9
}

default_options_train = {
    'init_lr': 5e-4,
    'stop_lr': 1e-6,
    'batch_size': 128,
    'epochs': 10000,

    'lr_scheduler': {
        'type': "validation",
        'nepochs': 50,
        'factor': 2,
        'decay_rate': 0.98,
        'decay_steps': 6000,
        'decay_min': 1e-7,
        'offset_steps': 5000,
    },
    'log_interval': 20,
    'grad_clip': 10000,
    'grad_clip_norm': 10000,

    'iw_samples': 1,
    'supersampling': 1,

    'annealing_options': {
        'init_annealing': 1.0,
        'scheduler':
            {
                'type': 'linear',
                'rate': 1e-4
            }
    },

    'freebits_options': {
        'init_threshold': 0.5,
        'grouping': "",
        'threshold_behaviour': "linear",
        'scheduler':
        {
            'type': 'exponential',
            'decay_rate': 0.5,
            'decay_steps': 300000
        }
    },
}

default_options_test = {
    'batch_size': 512,
    'iw_samples': 1
}



default_piano_layer_options = {
    'latent_size': 8,
    'size_multiplier': 8,
    'bypass_frac': 1.0,
    'ar_conn_frac': 1,
    'stride': 1,
    'inf_post_options': {'module': "SkipConnection1D"},
    'inf_pre_options':  {'module': "Wavenet", 'kernel_size': 2, 'dilation': (1, 1, 1, 1)},
    'gen_pre_options':  {'module': "Wavenet", 'kernel_size': 1, 'dilation': (1, 1)},
    'gen_post_options': {'module': "Wavenet", 'kernel_size': 1, 'dilation': (1, 1)},
    'ar_conn_options':  {'module': "SkipConnection1D"},
}


default_model_options_piano = {
    'vae_options': {
        'default_ar_layer': default_piano_layer_options,
        'ar_layers_options': [
            {},
            {'latent_size': 4, 'stride': 2},
            {'latent_size': 2, 'stride': 2},
            {'latent_size': 1, 'stride': 2},
            {'latent_size': 1, 'stride': 2}
        ],
        'default_fc_layer': {},
        'fc_layers_options': []
     },
    'pred_options': {"type": "bernoulli"},

    'nn_normalization': "",
    'data_normalize': True,

    "dimensions": 1,
}

default_model_layer_options_blizzard = {
    'latent_size': 1,
    'size': 26,
    'bypass_frac': 1.0,
    'ar_conn_frac': 1.0,
    'stride': 1,
    'inf_post_options': {'module': "SkipConnection1D"},
    'inf_pre_options':  {'module': "Wavenet", 'kernel_size': 2, 'dilation': (1, 1, 1, 1, 1)},
    'gen_pre_options':  {'module': "Wavenet", 'kernel_size': 1, 'dilation': (1, 1)},
    'gen_post_options': {'module': "ConditionedWavenet", 'kernel_size': 1, 'dilation': (1, 1)},
    'ar_conn_options':  {'module': "SkipConnection1D"},
}

default_model_options_blizzard = {
    'vae_options': {
        'default_ar_layer': default_model_layer_options_blizzard,
        'ar_layers_options': [
            {},
            {'latent_size': 5, 'size': 58, 'stride': 5},
            {'latent_size': 25, 'size': 128, 'stride': 5},
            {'latent_size': 64, 'size': 256, 'stride': 4},
            {'latent_size': 64, 'size': 256, 'stride': 1,
             'inf_pre_options':  {'module': "Wavenet", 'kernel_size': 2, 'dilation': (1, 2, 4, 8, 16)}},
        ],
        'default_fc_layer': {},
        'fc_layers_options': []
     },
    'pred_options': {"type": "normal_mixture", "n_components": 2},


    'nn_normalization': "",
    'data_normalize': False,

    "dimensions": 1,
}

default_model_layer_options_iamondb = {
    'latent_size': 8,
    'size_multiplier': 8,
    'bypass_frac': 1.0,
    'ar_conn_frac': 1,
    'stride': 1,
    'inf_post_options': {'module': "SkipConnection1D"},
    'inf_pre_options':  {'module': "Wavenet", 'kernel_size': 2, 'dilation': (1, 1, 1, 1)},
    'gen_pre_options':  {'module': "Wavenet", 'kernel_size': 1, 'dilation': (1, 1)},
    'gen_post_options': {'module': "Wavenet", 'kernel_size': 1, 'dilation': (1, 1)},
    'ar_conn_options':  {'module': "SkipConnection1D"},
}

default_model_options_iamondb = {

    'vae_options': {
        'default_ar_layer': default_model_layer_options_iamondb,
        'ar_layers_options': [
            {},
            {'latent_size': 8, 'stride': 2},
            {'latent_size': 8, 'stride': 2},
            {'latent_size': 4, 'stride': 2},
            {'latent_size': 2, 'stride': 1,
               'inf_pre_options':  {'module': "Wavenet", 'kernel_size': 2, 'dilation': (1, 2, 4, 8)}},
        ],
        'default_fc_layer': {},
        'fc_layers_options': []
     },
    'pred_options': {"type": "combination",
                     "sub_dists": [{"type": "bernoulli", "output_size": 1},
                                   {"type": "normal", "output_size": 2}]},


    'nn_normalization': "",
    'data_normalize': True,

    "dimensions": 1,
}

default_RMSprop_options = {
    'momentum': 0.9
}

default_Adam_options = {
    'weight_decay': 0.00,
    'eps': 1e-8,
}



default_options = {
    'cuda': False,
    'seed': 1234,
    'logdir': None,
    'run_name': None,
    'load_model': None,
    'save_code': True,
    'dataset_basepath':  '~/datasets',

    'll_normalization': "",
    'ind_latentstate': False,

    'train_options': default_options_train,
    'test_options': default_options_test,

    'dataset': "blizzard",
    'dataset_options': {},
    'pianoroll_options': default_pianoroll_options,
    'blizzard_options': default_blizzard_options,
    'iamondb_options': default_iamondb_options,
    'deepwriting_options': default_deepwriting_options,

    'model_options': default_model_options_blizzard,

    'optimizer': 'Adam',
    'optimizer_options': {},
    'RMSprop_options': default_RMSprop_options,
    'Adam_options': default_Adam_options,
}