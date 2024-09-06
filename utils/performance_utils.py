import os
import json
import numpy as np
import pickle as pkl
from utils.metrics import confusion_matrix_analysis
from utils.distributed_utils import logger


def get_ntrainparams(model):
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, trainable_params


def log_metrics(metrics, metric_type):
    scores_values_str = ','.join([f"{key} : {value}" for key, value in metrics.items()])
    logger.info(scores_values_str)


def log_params_flops(flops, macs, params):
    logger.info("Summary:\n+-----------+--------------+------------+\n"
                f"|{'flops (G)':^11}|{'macs (GMACs)':^14}|{'params (M)':^12}|\n"
                "+-----------+--------------+------------+\n"
                f"|{flops.split(' ')[0]:^11}|{macs.split(' ')[0]:^14}|{params.split(' ')[0]:^12}|\n"
                "+-----------+--------------+------------+")
