import numpy as np
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
from utils.metrics import IoU
from utils.distributed_utils import get_dist_info, logger


def run_iterate(model, data_loader, criterion, config, mode="train", device=None, dtype=torch.float32):

    model.train() if mode == "train" else model.eval()

    display_step = config.PIPELINE.train.display_step
    local_rank = config.local_rank

    iou_meter = IoU(
        num_classes=config.DATASET.num_classes,
        ignore_index=config.DATASET.ignore_index,
        cm_device=device,
    )

    t_start = time.time()
    for i, batch in enumerate(tqdm(data_loader) if local_rank == 0 else data_loader):
        if device is not None:
            batch = to_device(batch, device)
        x, y = batch

        if mode != "train":
            with torch.no_grad():
                out = model(x)
        else:
            out = model(x)

        loss = criterion(out.float(), y.long())

        if mode == "train":
            model.backward(loss)
            model.step()

        with torch.no_grad():
            pred = out.softmax(dim=1).argmax(dim=1)
        iou_meter.add(pred, y.long())

        if mode == "train" and (i + 1) % display_step == 0:
            scores = iou_meter.value()
            scores_values_str = ','.join([f"{key} : {value}" for key, value in scores.items()])
            logger.tqdm_write(scores_values_str)

    t_end = time.time()
    total_time = t_end - t_start
    scores = iou_meter.value()
    iou, miou, acc, oa = scores.values()

    metrics = {
        f"{mode}_iou": iou,
        f"{mode}_mIoU": miou,
        f"{mode}_accuracy": acc,
        f"{mode}_OA": oa,
        f"{mode}_loss": loss.item(),
        f"{mode}_epoch_time": total_time,
    }

    if mode == "test":
        return metrics, iou_meter.conf_metric.value()  # confusion matrix
    else:
        return metrics


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return [to_device(c, device) for c in x]


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()
