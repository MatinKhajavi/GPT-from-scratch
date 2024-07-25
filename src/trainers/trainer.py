
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class Trainer:

    def __init__(self) -> None:
        pass

    
    def train(self):
        pass

    @torch.no_grad()
    def validate(self):
        pass

    @torch.no_grad()
    def evaluate(self):
        pass

    def get_lr(self):
        pass

    def get_optimizer(self):
        pass