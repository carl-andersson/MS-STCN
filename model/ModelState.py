import torch
import torch.optim as optim
import os.path


class ModelState:
    """
    Container for the model and optimizer

    model
    optimizer
    """

    def __init__(self, seed, optimizer, optimizer_options, init_lr, model, init_annealing, init_freebits):
        torch.manual_seed(seed)

        self.model = model

        # Optimization parameters
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=init_lr, **optimizer_options)
        self.posterior_optimizer = getattr(optim, optimizer)(self.model.vae.posterior_parameters(), lr=init_lr, **optimizer_options)

        self.annealing = init_annealing
        self.freebits = init_freebits



    def load_model(self, path, name='model.pt'):
        file = path if os.path.isfile(path) else os.path.join(path, name)
        try:
            ckpt = torch.load(file, map_location=lambda storage, loc: storage)
        except NotADirectoryError:
            raise Exception("Could not find model: " + file)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.freebits = ckpt["freebits"]
        self.annealing = ckpt["annealing"]
        epoch = ckpt['epoch']
        return epoch

    def save_model(self, epoch, vloss, elapsed_time,  path, name='model.pt'):
        torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'vloss': vloss,
                'elapsed_time': elapsed_time,
                'freebits': self.freebits,
                'annealing': self.annealing
            },
            os.path.join(path, name))
