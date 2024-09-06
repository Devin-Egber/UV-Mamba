import io
import os
import json
import pickle as pkl
import numpy as np


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)


def checkpoint(log, config):
    for key, value in log.items():
        for k, v in value.items():
            if isinstance(v, np.ndarray):
                log[key][k] = v.tolist()

    with open(os.path.join(config.res_dir, "trainlog.json"), "w") as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, conf_mat, config):

    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()

    with open(os.path.join(config.res_dir, "test_metrics.json"), "w") as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(
        conf_mat,
        open(
            os.path.join(config.res_dir, "conf_mat.pkl"), "wb"
        ),
    )


def read_ds_config(config_path, mode="r"):
    if not isinstance(config_path, io.IOBase):
        config_path = open(config_path, mode=mode)

    config = json.load(config_path)
    config_path.close()

    return config
