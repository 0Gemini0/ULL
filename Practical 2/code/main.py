#!/usr/bin/env python3
import os.path as osp
from time import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SparseAdam
from torch.autograd import Variable
import msgpack
import numpy as np

from settings import parse_settings
from dataset import SkipGramData, EmbedAlignData, sort_collate
from models.skipgram import SkipGram
from models.bayesian import Bayesian
from models.embedalign import EmbedAlign


def construct_data_path(opt, name):
    return osp.join(opt.data_path, opt.dataset, opt.training_test + "_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                    + "_" + str(opt.window_size) + "_" + str(opt.k) + "_" + name + "." + opt.language)


def construct_data_path_ea(opt, name):
    return osp.join(opt.data_path, opt.dataset, opt.training_test + "_" + str(bool(opt.lowercase)) + "_" + str(opt.max_sentence_size) + "_" + str(opt.vocab_size) + "_" + name)


def construct_model_path(opt, is_best):
    if is_best:
        return osp.join(opt.out_path, opt.dataset, opt.model + "_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                        + "_" + str(opt.window_size) + "_" + str(opt.k) + ".pt")
    else:
        return osp.join(opt.out_path, opt.dataset, opt.model + "_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                        + "_" + str(opt.window_size) + "_" + str(opt.k) + "_checkpoint.pt")


def save_checkpoint(opt, model, optimizers, epoch, loss, is_best):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'loss': loss,
        'optimizers': [optimizer.state_dict() for optimizer in optimizers],
    }
    torch.save(checkpoint, construct_model_path(opt, is_best))


def load_checkpoint(opt, model, optimizers):
    checkpoint = torch.load(construct_model_path(opt, False))
    opt.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizers = [optimizers[i].load_state_dict(checkpoint['optimizers'][i]) for i in range(len(optimizer))]
    loss = checkpoint["loss"]
    return opt, model, optimizers, loss


def main(opt):
    # We activate the GPU if cuda is available, otherwise computation will be done on cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Using device: {}".format(device))

    # Now we load the data fitting the selected model
    print("Loading Data...")
    if opt.model == "embedalign":
        data = DataLoader(EmbedAlignData(construct_data_path_ea(opt, "data.both"), opt.v_dim - 1),
                          batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sort_collate)
    else:
        data = DataLoader(SkipGramData(construct_data_path(opt, "samples"), construct_data_path(opt, "negativeSamples"),
                                       opt.v_dim - 1), batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print("Data was succesfully loaded.")

    # We load the selected model and place it on the available device(s)
    if opt.model == "skipgram":
        model = SkipGram(opt.v_dim, opt.d_dim, opt.v_dim-1)
    elif opt.model == "bayesian":
        model = Bayesian(opt.v_dim, opt.d_dim, opt.h_dim, opt.v_dim-1)
    elif opt.model == "embedalign":
        model = EmbedAlign(opt.v_dim, opt.v_dim, opt.d_dim, opt.h_dim, opt.neg_dim, opt.v_dim-1, device)
    else:
        raise Exception("Model not recognized, choose [skipgram, bayesian, embedalign]")

    if opt.parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        print("Using parallel processing on GPUs")
    else:
        model.to(device)

    # To work with ADAM and Sparse Embeddings, we need (retardedly) to manually store sparse parameters
    parameters_sparse = list(filter(lambda p: p.requires_grad, model.sparse_params))
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    parameters = [p for p in parameters if not any(p is p_ for p_ in parameters_sparse)]
    optimizers = []
    if parameters_sparse:
        optimizers.append(SparseAdam(parameters_sparse, opt.lr))
    if parameters:
        optimizers.append(Adam(parameters, opt.lr))

    losses = []
    best_loss = np.inf
    for i in range(opt.num_epochs):
        ep_loss = 0.
        t = 0.
        for j, data_in in enumerate(data):
            start = time()
            # No longer tedious! Send data to selected device
            data_in = [inp.to(device) for inp in data_in]

            # Actual training
            loss = model(data_in)

            ep_loss += loss.item()

            # Get gradients and update parameters
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            # See progress
            if j % 10 == 0:
                print("\rSteps this epoch: {}, time: {}s, avg_loss: {}".format(
                    j, t, ep_loss/j), end="", flush=True)

            t += time() - start

        avg_loss = ep_loss/j
        if avg_loss < best_loss:
            best_loss = avg_loss
            is_best = True
        losses.append(avg_loss)

        # Save_checkpoint
        save_checkpoint(opt, model, optimizers, i, losses, is_best)
        print("Epoch: {}, Average Loss: {}".format(i, avg_loss))
        is_best = False


if __name__ == '__main__':
    opt = parse_settings()
    main(opt)
