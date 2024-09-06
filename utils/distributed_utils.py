import deepspeed
import logging
import torch
import torch.distributed as dist
from tqdm import tqdm


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val) if not isinstance(val, torch.Tensor) else val

    if not isinstance(val, torch.Tensor):
        t = torch.tensor(val, device="cuda")
    else:
        t = val.clone().detach()
    dist.barrier()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def get_dist_info():
    return dist.get_rank(), dist.get_world_size()


def all_metrics_reduce(value, device, world_size):
    if world_size > 1:
        value_total = torch.tensor([value], device=device)
        dist.all_reduce(tensor=value_total, op=dist.ReduceOp.SUM)
        value_mean = value_total / world_size
        return value_mean.item()
    return value


def dist_barrier(world_size):
    if world_size > 1:
        dist.barrier()


class RankLogger(object):
    def __init__(self, rank, name):
        self.rank = rank
        self.logger = logging.getLogger(name)

    def info(self, message):
        if self.rank == 0:
            self.logger.info(message)

    def warning(self, message):
        if self.rank == 0:
            self.logger.warning(message)

    def error(self, message):
        if self.rank == 0:
            self.logger.error(message)

    def debug(self, message):
        if self.rank == 0:
            self.logger.debug(message)

    def tqdm_write(self, message):
        if self.rank == 0:
            tqdm.write(message)


logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
deepspeed.init_distributed(dist_backend="nccl")
rank, _ = get_dist_info()
logger = RankLogger(rank, f"{__name__.split('.')[-1]}.py")
