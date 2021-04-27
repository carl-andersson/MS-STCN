# %% Initialization
import torch.nn as nn
from model.base import logexpmean, calc_ll_estimate
import numpy as np
import torch
import time
import logger
import sys


def run_train(start_epoch, cuda, ll_normalization, ind_latentstate, modelstate, logdir, loaders, train_options, test_options):

    def validate(loader, iw_samples, training_cost=False):
        modelstate.model.eval()
        total_vloss = 0
        n_tot = 0
        with torch.no_grad():
            for i, (x, y, l) in enumerate(loader):
                if cuda:
                    x = x.cuda()
                    if l is not None:
                        l = l.cuda()
                if training_cost:
                    vloss = calc_ll_estimate(x, *modelstate.model(x, lengths=l, iw_samples=iw_samples), l, iw_samples,
                                             modelstate.annealing, modelstate.freebits, train_options["freebits_options"],
                                             ll_normalization=ll_normalization, ind_latentstate=ind_latentstate)
                else:
                    vloss = calc_ll_estimate(x, *modelstate.model(x, lengths=l, iw_samples=iw_samples), l, iw_samples,
                                             1.0, 0, None,
                                             ll_normalization=ll_normalization, ind_latentstate=ind_latentstate)

                batchsize = x.size()[0]
                n_tot += batchsize

                total_vloss += batchsize * vloss.item()

        return total_vloss / n_tot

    def train(epoch):
        modelstate.model.train()
        total_loss = 0
        n_tot = 0
        for i, (x, y, l) in enumerate(loaders["train"]):
            if cuda:
                x = x.cuda()
                if l is not None:
                    l = l.cuda()
            # Exponential learning rate
            if train_options["lr_scheduler"]["type"] == "exponential":
                lr = modelstate.optimizer.param_groups[0]["lr"]
                lr = lr*train_options["lr_scheduler"]["decay_rate"] ** \
                         (1.0 / train_options["lr_scheduler"]["decay_steps"])
                if train_options["lr_scheduler"]["decay_min"] < lr:
                    for param_group in modelstate.optimizer.param_groups:
                        param_group['lr'] = lr

            # Exponential decaying freebits
            if train_options["freebits_options"]["scheduler"]["type"] == "exponential":
                modelstate.freebits = modelstate.freebits * \
                                     train_options["freebits_options"]["scheduler"]["decay_rate"] ** \
                                     (1.0 / train_options["freebits_options"]["scheduler"]["decay_steps"])


            if train_options["annealing_options"]["scheduler"]["type"] == "linear":
                modelstate.annealing = np.minimum(1.0, modelstate.annealing + 1.0/train_options["annealing_options"]["scheduler"]["rate"])



            if train_options["supersampling"] > 1:
                x = x.repeat(train_options["supersampling"], *[1] * len(x.size()[1:]))

            modelstate.optimizer.zero_grad()
            loss = calc_ll_estimate(x, *modelstate.model(x, lengths=l, iw_samples=train_options["iw_samples"]), l, train_options["iw_samples"],
                                    modelstate.annealing, modelstate.freebits,
                                    train_options["freebits_options"], ll_normalization=ll_normalization,
                                    ind_latentstate=ind_latentstate)
            loss.backward()
            nn.utils.clip_grad_value_(modelstate.model.parameters(), train_options["grad_clip"])
            nn.utils.clip_grad_norm_(modelstate.model.parameters(), train_options["grad_clip_norm"])

            modelstate.optimizer.step()

            batchsize = x.size()[0]
            n_tot += batchsize
            total_loss += batchsize * loss.item()

            # Extract learning rate
            lr = modelstate.optimizer.param_groups[0]["lr"]

            if i % train_options['log_interval'] == 0:
                print('Train Epoch: {:5d} [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.6f}\tLoss: {:.6f}'.format(
                    epoch, i, len(loaders["train"]), 100. * i / len(loaders["train"]), lr, total_loss / n_tot))
                n_tot = 0
                total_loss = 0


    def log(epoch, loss, loss_w_fb, vloss, vll, trainelbo, velbo, lr, freebits, annealing):
        print('-' * 100)
        print('Train Epoch: {:5d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.6f}'
              '\tLoss: {:.6f}\tVal Loss: {:.6f}'.
              format(epoch, len(loaders["train"]), len(loaders["train"]), 100., lr, loss, vloss))
        print('-' * 100)

        logger.log(epoch, {'train/cost': loss, 'train/cost_w_freebits': loss_w_fb,
                           'train/lr': lr, 'train/freebits': freebits,
                           'train/annealing': annealing, 'train/elbo': trainelbo,
                           'valid/cost': vloss, 'valid/ll': vll, 'valid/elbo': velbo})



    if torch.cuda.is_available():
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")



    try:
        # Train
        epoch = start_epoch
        vloss = validate(loaders["valid"], train_options["iw_samples"])
        all_vlosses = []
        best_vloss = vloss
        start_time = time.clock()
        for epoch in range(start_epoch, start_epoch + train_options["epochs"]+1):
            # Train and validate
            train(epoch)

            vloss = validate(loaders["valid"], train_options["iw_samples"])
            loss_w_freebits = validate(loaders["train_eval"], train_options["iw_samples"], training_cost=True)
            loss = validate(loaders["train_eval"], train_options["iw_samples"])
            vll = validate(loaders["valid"], test_options["iw_samples"])
            velbo = validate(loaders["valid"], 1)
            trainelbo = validate(loaders["train_eval"], 1)

            # Save losses
            all_vlosses += [vloss]
            if vloss < best_vloss:
                best_vloss = vloss
                modelstate.save_model(epoch, best_vloss, time.clock() - start_time, logdir, 'best_model.pt')
                logger.register_summary({'best_validation_loss': best_vloss, 'best_validation_loss_epoch': epoch})


            # Extract learning rate
            lr = modelstate.optimizer.param_groups[0]["lr"]

            # Print validation results
            log(epoch, loss, loss_w_freebits, vloss, vll, trainelbo, velbo, lr, modelstate.freebits, modelstate.annealing)

            # Validation dependent learning rate
            if train_options['lr_scheduler']["type"] == "validation":

                if len(all_vlosses) > train_options['lr_scheduler']['nepochs'] and \
                        min(all_vlosses[-train_options['lr_scheduler']['nepochs']:]) > \
                        min(all_vlosses[:-train_options['lr_scheduler']['nepochs']]):
                    all_vlosses = []
                    lr = lr / train_options['lr_scheduler']['factor']
                    for param_group in modelstate.optimizer.param_groups:
                        param_group['lr'] = lr

            # Early stoping
            if lr < train_options['stop_lr']:
                print('-' * 89)
                print('Training finished: Minimal learning rate reached')
                print('-' * 89)
                break

        modelstate.save_model(epoch, vloss, time.clock() - start_time, logdir, 'final_model.pt')

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        modelstate.save_model(epoch, vloss, time.clock() - start_time, logdir, 'interrupted_model.pt')
        print('-' * 89)


    logger.finish()
    sys.stdout.flush()
    logger.remove_redirects()