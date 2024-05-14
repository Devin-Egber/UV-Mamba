import argparse
import torch
import yaml
from addict import Dict
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

from utils.model_utils import *
from utils.performance_utils import log_params_flops


def utae_input_constructor(x_shape, batch_positions_shape, gt_shape=None,
                           is_diffusion_based=False, device=None):
    x = torch.rand(*x_shape).to(device)
    dates = torch.rand(*batch_positions_shape).to(device)
    if gt_shape is not None:
        gt = torch.ones(*gt_shape).long().to(device)
    inputs = {
        "x": x,
        "batch_positions": dates,
        "gt": gt if gt_shape is not None else None,
        "is_diffusion_based": is_diffusion_based
    }
    return inputs


def input_constructor(x_shape, batch_positions_shape=None, img_mask_shape=None, date_pos_shape=None,temporal_mask_shape=None, gt_shape=None,
                           is_diffusion_based=False, device=None):
    x = torch.rand(*x_shape).to(device)
    if batch_positions_shape is not None:
        batch_positions = torch.rand(*batch_positions_shape).to(device)
    if img_mask_shape is not None:
        img_mask = torch.rand(*img_mask_shape).to(device)
    if date_pos_shape is not None:
        date_pos = torch.rand(*date_pos_shape).to(device)
    if temporal_mask_shape is not None:
        temporal_mask = torch.rand(*temporal_mask_shape).to(device)

    if gt_shape is not None:
        gt = torch.ones(*gt_shape).long().to(device)
    inputs = {
        "x": x,
        "batch_positions": batch_positions if batch_positions_shape is not None else None,
        "img_mask": img_mask if img_mask_shape is not None else None,
        "date_pos": date_pos if date_pos_shape is not None else None,
        "temporal_mask": temporal_mask if temporal_mask_shape is not None else None,
        "gt": gt if gt_shape is not None else None,
        "is_diffusion_based": is_diffusion_based
    }
    return inputs


def get_model_input(config):
    device = config.local_rank
    backbone = config.BACKBONE
    model = get_model(config)

    x_shape, batch_positions_shape, img_mask_shape, date_pos_shape, temporal_mask_shape, gt_shape = None, None, None, None, None, None

    if backbone in ["utae"]:
        x_shape, batch_positions_shape = (1, 30, 10, 128, 128), (1, 30)
        if config.DIFFUSION:
            gt_shape = (1, 128, 128)
            model.diff_encoder.set_noise_schedule(device=device)

    elif backbone in ["exchanger_unet"]:
        x_shape, img_mask_shape, date_pos_shape, temporal_mask_shape = (1, 10, 30, 32, 32), (1, 30, 32, 32), (1, 30), (1, 30)
        if config.DIFFUSION:
            gt_shape = (1, 32, 32)
            model.diff_encoder.set_noise_schedule(device=device)

    elif backbone in ["exchanger_maskformer"]:
        x_shape, img_mask_shape, date_pos_shape, temporal_mask_shape = (1, 10, 30, 64, 64), (1, 30, 64, 64), (1, 30), (1, 30)
        if config.DIFFUSION:
            gt_shape = (1, 64, 64)
            model.diff_encoder.set_noise_schedule(device=device)

    elif backbone in ["TSViT"]:
        x_shape = (1, 30, 24, 24, 11)
        if config.DIFFUSION:
            gt_shape = (1, 24, 24)
            model.diff_encoder.set_noise_schedule(device=device)
    else:
        raise NotImplementedError

    inputs_shape = {
        "x_shape": x_shape,
        "batch_positions_shape": batch_positions_shape,
        "img_mask_shape": img_mask_shape,
        "date_pos_shape": date_pos_shape,
        "temporal_mask_shape": temporal_mask_shape,
        "gt_shape": gt_shape,
        "is_diffusion_based": config.DIFFUSION
    }

    return model, inputs_shape


def get_model_complexity_info(config):
    with get_accelerator().device(config.local_rank):
        device = config.local_rank
        model, inputs_shape = get_model_input(config)

        flops, macs, params = get_model_profile(model=model.to(device),
                                                kwargs=input_constructor(x_shape=inputs_shape['x_shape'],
                                                                              batch_positions_shape=inputs_shape['batch_positions_shape'],
                                                                              img_mask_shape=inputs_shape['img_mask_shape'],
                                                                              date_pos_shape=inputs_shape['date_pos_shape'],
                                                                              temporal_mask_shape=inputs_shape['temporal_mask_shape'],
                                                                              gt_shape=inputs_shape['gt_shape'],
                                                                              is_diffusion_based=inputs_shape['is_diffusion_based'],
                                                                              device=device),
                                                # dictionary of keyword arguments to the model.
                                                print_profile=True,
                                                # prints the model graph with the measured profile attached to each module
                                                detailed=True,  # print the detailed profile
                                                module_depth=-1,
                                                # depth into the nested modules, with -1 being the inner most modules
                                                top_modules=1,  # the number of top modules to print aggregated profile
                                                warm_up=10,
                                                # the number of warm-ups before measuring the time of each module
                                                as_string=True,
                                                # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                                output_file=None,
                                                # path to the output file. If None, the profiler prints to stdout.
                                                ignore_modules=None)  # the list of modules to ignore in the profiling

        log_params_flops(flops, macs, params)








def get_utae_model_complexity_info(config):
    with get_accelerator().device(config.local_rank):
        device = config.local_rank
        model = get_model(config)
        x_shape, batch_positions_shape = (1, 30, 10, 128, 128), (1, 30)
        gt_shape = None
        if config.DIFFUSION:
            gt_shape = (1, 128, 128)
            model.diff_encoder.set_noise_schedule(device=device)
        flops, macs, params = get_model_profile(model=model.to(config.local_rank),
                                                kwargs=utae_input_constructor(x_shape=x_shape,
                                                                              batch_positions_shape=batch_positions_shape,
                                                                              gt_shape=gt_shape,
                                                                              is_diffusion_based=config.DIFFUSION,
                                                                              device=device),
                                                # dictionary of keyword arguments to the model.
                                                print_profile=True,
                                                # prints the model graph with the measured profile attached to each module
                                                detailed=True,  # print the detailed profile
                                                module_depth=-1,
                                                # depth into the nested modules, with -1 being the inner most modules
                                                top_modules=1,  # the number of top modules to print aggregated profile
                                                warm_up=10,
                                                # the number of warm-ups before measuring the time of each module
                                                as_string=True,
                                                # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                                output_file=None,
                                                # path to the output file. If None, the profiler prints to stdout.
                                                ignore_modules=None)  # the list of modules to ignore in the profiling

        log_params_flops(flops, macs, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Configuration (.json) file to use')
    parser.add_argument('--local_rank', type=int, default=-1, help='Specifying the default GPU')
    config = parser.parse_args()

    with open(config.config_file, 'r') as config_file:
        model_config = yaml.safe_load(config_file)

    config = Dict({**model_config, **vars(config)})
    config = argparse.Namespace(**config)

    get_model_complexity_info(config)
