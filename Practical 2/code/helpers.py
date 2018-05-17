"""Additional functions to help accross scripts."""
import os.path as osp
import torch


def construct_data_path(opt, name):
    return osp.join(opt.data_path, opt.dataset, opt.training_test + "_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                    + "_" + str(opt.window_size) + "_" + str(opt.k) + "_" + name + opt.language)


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


def load_model(path, model):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    return model
