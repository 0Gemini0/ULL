#!/usr/bin/python3
import os.path as osp

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import msgpack
import numpy as np

from settings import parse_settings
from dataset import SkipGramData
from models.skipgram import SkipGram
from models.bayesian import Bayesian


def construct_data_path(opt, name):
    return osp.join(opt.data_path, opt.dataset, "training_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                    + "_" + str(opt.window_size) + "_" + str(opt.k) + "_" + name + "." + opt.language)


def construct_model_path(opt, is_best):
    if is_best:
        return osp.join(opt.out_path, opt.dataset, opt.model + "_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                        + "_" + str(opt.window_size) + "_" + str(opt.k) + ".pt")
    else:
        return osp.join(opt.out_path, opt.dataset, opt.model + "_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                        + "_" + str(opt.window_size) + "_" + str(opt.k) + "_checkpoint.pt")


def save_checkpoint(opt, model, optimizer, epoch, loss, is_best):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'loss': loss,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, construct_model_path(opt, is_best))


def load_checkpoint(opt, model, optimizer):
    checkpoint = torch.load(construct_model_path(opt, False))
    opt.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint["loss"]
    return opt, model, optimizer, loss


def main(opt):
    # Cuda functionality
    if opt.cuda is not None:
        if opt.cuda > -1:
            torch.cuda.set_device(opt.cuda)
        opt.cuda = True
    else:
        opt.cuda = False

    # Load data
    print("Loading Data...")
    data = DataLoader(SkipGramData(construct_data_path(opt, "samples"), construct_data_path(opt, "negativeSamples"), opt.v_dim-1),
                      batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    idx_to_word = msgpack.load(open(construct_data_path(opt, "indexWordMap"), 'rb'), encoding='utf-8')
    print("Data was succesfully loaded.")

    # Load model
    if opt.model == "skipgram":
        model = SkipGram(opt.v_dim, opt.d_dim, opt.v_dim-1)
    elif opt.model == "bayesian":
        model = Bayesian(opt.v_dim, opt.d_dim, opt.h_dim, opt.v_dim-1)
    elif opt.model == "embedalign":
        # do something else else
        raise NotImplementedError()
    else:
        raise Exception("Model not recognized, choose [skipgram, bayesian, embedalign]")

    if opt.cuda:
        model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Define optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(parameters, opt.lr)

    # Training loop
    losses = []
    best_loss = np.inf
    for i in range(opt.num_epochs):
        ep_loss = 0.
        for j, (center, pos_context, pos_mask, neg_context, neg_mask) in enumerate(data):
            # Tedious wrapping of Tensors into Variables that may or may not be cuda'd
            center = Variable(center)
            pos_context = Variable(pos_context)
            pos_mask = Variable(pos_mask)
            neg_context = Variable(neg_context)
            neg_mask = Variable(neg_mask)
            if opt.cuda:
                center = center.cuda()
                pos_context = pos_context.cuda()
                pos_mask = pos_mask.cuda()
                neg_context = neg_context.cuda()
                neg_mask = neg_mask.cuda()

            # Actual training
            loss = torch.sum(model(center, pos_context, pos_mask, neg_context, neg_mask))
            ep_loss += loss.data[0]

            # Get gradients and update parameters
            loss.backward()
            optimizer.step()

            # See Batch Loss
            print("\rBatch Loss: {}".format(loss.data[0]), end="", flush=True)

            # See progress
            if j % 1000 == 0:
                print("\rSteps this epoch: {}".format(j), end="", flush=True)
        if ep_loss < best_loss:
            best_loss = ep_loss
            is_best = True
        losses.append(ep_loss)

        # Save_checkpoint
        save_checkpoint(opt, model, optimizer, i, losses, is_best)

        print("Epoch: {}, Average Loss: {}".format(i, ep_loss/j))

        is_best = False


if __name__ == '__main__':
    opt = parse_settings()
    main(opt)
