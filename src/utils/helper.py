import os
import torch


def probs_to_label(probs, is_binary=True):
    if is_binary:
        return (probs > 0.5).astype(float)
    else:
        return probs.argmax(axis=-1)


def create_dirs(dirs):

    if not os.path.exists(dirs):
        os.makedirs(dirs)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
