#!/usr/bin/python3
import os.path as osp

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import msgpack

from settings import parse_settings
from dataset import SkipGramData
from models.skipgram import SkipGram
from models.bayesian import Bayesian


def construct_data_path(opt, name):
    return osp.join(opt.data_path, opt.dataset, "training_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                    + "_" + str(opt.window_size) + "_" + str(opt.k) + "_" + name + "." + opt.language)


def construct_model_path(opt):
    return osp.join(opt.data_path, opt.dataset, opt.model + "_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                    + "_" + str(opt.window_size) + "_" + str(opt.k) + ".pt")


def main(opt):
    # Cuda functionality
    if opt.cuda is not None and opt.cuda > -1:
        torch.cuda.set_device(opt.cuda)
        opt.cuda = True

    # Load data
    data = DataLoader(SkipGramData(construct_data_path(opt, "samples"), construct_data_path(opt, "negativeSamples"), opt.v_dim-1),
                      batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    idx_to_word = msgpack.load(open(construct_data_path(opt, "indexWordMap"), 'rb'), encoding='utf-8')

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

    # Define optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(parameters, opt.lr)

    # Training loop
    for i in range(opt.num_epochs):
        ep_loss = 0.
        for j, (center, pos_context, pos_mask, neg_context, neg_mask) in enumerate(data):
            # Tedious wrapping of Tensors into Variables that may or may not be cuda'd
            print(torch.max(pos_context))
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
            loss = model(center, pos_context, pos_mask, neg_context, neg_mask)
            ep_loss += loss.data()

            # Get gradients and update parameters
            loss.backward()
            optimizer.step()

        print("Epoch: {}, Average Loss: {}".format(i, ep_loss/j))

    # Save model with training settings in the name
    model.save_state_dict(construct_model_path(opt))


if __name__ == '__main__':
    opt = parse_settings()
    main(opt)
