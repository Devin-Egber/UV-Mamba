import argparse
import warnings
import torch
import yaml
from addict import Dict

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

from utils.model_utils import *
from utils.model_utils import get_model

from utils.performance_utils import log_params_flops


def get_model_complexity_info(config):
    with get_accelerator().device(config.local_rank):
        device = config.local_rank
        model = get_model(config)

        flops, macs, params = get_model_profile(model=model.to(device),
                                                input_shape=(1, 3, 1024, 1024),
                                                args=None,
                                                kwargs=None,
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
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        default="config/uv/uv_mamba/deform_uvmamba_cityscapes.yaml",
                        help='Configuration (.json) file to use')

    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Specifying the default GPU')

    config = parser.parse_args()

    with open(config.config_file, 'r') as config_file:
        model_config = yaml.safe_load(config_file)

    config = Dict({**model_config, **vars(config)})
    config = argparse.Namespace(**config)

    get_model_complexity_info(config)
