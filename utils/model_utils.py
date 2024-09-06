import os
import shutil
import torch
from pathlib import Path
from torch import nn
from torch.nn import init

from utils.distributed_utils import logger


def get_model(config):

    if config.BACKBONE == "uvmamba":
        from models.model import UVMamba
        base_model = UVMamba(config)

    elif config.BACKBONE == "uvmamba_no_deform":
        from models.model import UVMambaNoDeform
        base_model = UVMambaNoDeform(config)

    elif config.BACKBONE == "uvmamba_no_ssm":
        from models.model import UVMambaNoSSM
        base_model = UVMambaNoSSM(config)

    elif config.BACKBONE == "uvmamba_parallel":
        from models.model import UVMambaParallel
        base_model = UVMambaParallel(config)

    elif config.BACKBONE == "uvmamba_reverse":
        from models.model import UVMambaReverse
        base_model = UVMambaReverse(config)

    else:
        raise NotImplementedError

    return base_model


def init_weights(module):
    '''
        Initializes a model's parameters.
        Credits to: https://gist.github.com/jeasinema

        Usage:
            model = Model()
            model.apply(weight_init)
        '''
    if isinstance(module, nn.Conv1d):
        init.normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.Conv2d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.Conv3d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.ConvTranspose1d):
        init.normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.ConvTranspose2d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.ConvTranspose3d):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            init.normal_(module.bias.data)
    elif isinstance(module, nn.BatchNorm1d):
        init.normal_(module.weight.data, mean=0, std=1)
        init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.normal_(module.weight.data, mean=0, std=1)
        init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm3d):
        init.normal_(module.weight.data, mean=0, std=1)
        init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight.data)
        try:
            init.normal_(module.bias.data)
        except AttributeError:
            pass
    elif isinstance(module, nn.LSTM):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(module, nn.LSTMCell):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(module, nn.GRU):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(module, nn.GRUCell):
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def load_weights(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)["module"], strict=False)


def save_state_dict(model, path, save_total_limit):
    assert save_total_limit > 0, "save_total_limit must be greater than 0"
    model.save_checkpoint(path)
    glob_checkpoints = [str(x) for x in Path(path).glob(f"global_step*") if os.path.isdir(x)]
    glob_checkpoints_sorted = sorted(glob_checkpoints, key=lambda x: int(x.split("global_step")[-1]))

    if len(glob_checkpoints_sorted) > save_total_limit:
        logger.info(f"Deleting older checkpoint({glob_checkpoints_sorted[0]}) due to args.save_total_limit")
        shutil.rmtree(glob_checkpoints_sorted[0], ignore_errors=True)


def resume_state_dict(model, checkpoint_path):
    model.load_checkpoint(checkpoint_path)
    glob_checkpoints = [str(x) for x in Path(checkpoint_path).glob(f"global_step*") if os.path.isdir(x)]
    glob_checkpoints_sorted = sorted(glob_checkpoints, key=lambda x: int(x.split("global_step")[-1]), reverse=True)
    resume_steps = glob_checkpoints_sorted[0].split("global_step")[1]
    return int(resume_steps)

