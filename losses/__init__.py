import torch
from torch import nn


def get_loss(dataset_config):

    if dataset_config.dataset == 'uvseg':
        criterion = nn.CrossEntropyLoss()
    elif dataset_config.dataset == 'cityscapes':
        criterion = nn.CrossEntropyLoss(ignore_index=dataset_config.ignore_index)
    else:
        raise NotImplementedError

    return criterion