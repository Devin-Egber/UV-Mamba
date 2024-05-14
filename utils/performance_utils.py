import os
import json
import numpy as np
import pickle as pkl
from utils.metrics import confusion_matrix_analysis
from utils.distributed_utils import logger


def overall_performance(config):
    num_classes = config.DATASET.num_classes
    res_dir = config.PATH.res_dir
    cm = np.zeros((num_classes, num_classes))
    for fold in range(1, 6):
        cm += pkl.load(
            open(
                os.path.join(res_dir, "Fold_{}".format(fold), "conf_mat.pkl"),
                "rb",
            )
        )

    if config.DATASET.avoid_index is not None:
        avoid_index = config.DATASET.avoid_index
        cm = np.delete(cm, avoid_index, axis=0)
        cm = np.delete(cm, avoid_index, axis=1)

    _, perf = confusion_matrix_analysis(cm)
    metrics = {'test_accuracy': perf["Accuracy"] * 100, 'test_mIoU': perf["MACRO_IoU"] * 100}

    log_metrics(metrics, 'test')

    with open(os.path.join(res_dir, "overall.json"), "w") as file:
        file.write(json.dumps(perf, indent=4))


def get_ntrainparams(model):
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, trainable_params


def log_metrics(metrics, metric_type):
    scores_values_str = ','.join([f"{key} : {value}" for key, value in metrics.items()])
    logger.info(scores_values_str)
    # logger.info("Summary:\n+---------+----------+\n"
    #             f"|{'Acc':^9}|{'mIoU':^10}|\n"
    #             "+---------+----------+\n"
    #             f"|{metrics[metric_type + '_accuracy']:^9.2f}|{metrics[metric_type + '_mIoU']:^10.2f}|\n"
    #             "+---------+----------+")


def log_params_flops(flops, macs, params):
    logger.info("Summary:\n+-----------+--------------+------------+\n"
                f"|{'flops (G)':^11}|{'macs (GMACs)':^14}|{'params (M)':^12}|\n"
                "+-----------+--------------+------------+\n"
                f"|{flops.split(' ')[0]:^11}|{macs.split(' ')[0]:^14}|{params.split(' ')[0]:^12}|\n"
                "+-----------+--------------+------------+")
