# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
import sys
import os


def set_redirects(logdir):
    sys.stdout = RedirectLogger(logdir, sys.stdout)
    sys.stderr = RedirectLogger(logdir, sys.stderr)

def remove_redirects():
    sys.stdout = sys.stdout.terminal
    sys.stderr = sys.stderr.terminal

class RedirectLogger(object):
    def __init__(self, logdir, term):
        self.terminal = term
        self.log = open(os.path.join(logdir, "run.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()




class TensorboardWrapper:

    def __init__(self, logdir):
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=logdir)

    def log(self, epoch, log_dict):
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, epoch)

    def register_summary(self, summary_dict):
        pass

    def finish(self):
        pass

class WandbWrapper:
    def __init__(self, project, run_name, options={}):
        import wandb
        self.writer = wandb
        self.writer.init(project=project, name=run_name, config=options)

    def log(self, epoch, log_dict):
        self.writer.log({'epoch': epoch, **log_dict})

    def register_summary(self, summary_dict):
        for k,v in summary_dict.items():
            self.writer.run.summary[k] = v

    def finish(self):
        self.writer.finish()


_loggers = []


def init(use_wandb, use_tensorboard, kwargs_wandb={}, kwargs_tensorboard={}):
    _loggers.clear()
    if use_wandb:
        _loggers.append(WandbWrapper(**kwargs_wandb))
    if use_tensorboard:
        _loggers.append(TensorboardWrapper(**kwargs_tensorboard))


def log(epoch, log_dict):
    for logger in _loggers:
        logger.log(epoch, log_dict)


def register_summary(summary_dict):
    for logger in _loggers:
        logger.register_summary(summary_dict)


def finish():
    for logger in _loggers:
        logger.finish()

