import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import math
from typing import Optional, Any, Union
from src.dataset.dataloader import DataLoader

class Trainer:
    """
    Trainer class for managing the training process.

    :param model: The model to be trained.
    :param train_loader: The data loader for the training dataset.
    :param val_loader: The data loader for the validation dataset.
    :param raw_model: The original neural network model, unwrapped from DDP if applicable.
    :param warmup_iters: Number of iterations for learning rate warmup.
    :param max_iters: Total number of iterations for training.
    :param max_lr: Maximum learning rate.
    :param min_lr: Minimum learning rate.
    :param use_ddp: Flag to setup DDP or not.
    :param monitor: Flag to monitor the training process.
    :param log_dir: Directory to store logs.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 raw_model: Optional[torch.nn.Module] = None,
                 warmup_iters: int = 715,
                 max_iters: int = 19073,
                 max_lr: float = 6e-4,
                 min_lr: float = 6e-3,
                 use_ddp: bool = False,
                 monitor: bool = True,
                 log_dir: str = "log") -> None:
        
        self.model = model
        self.raw_model = raw_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.use_ddp = use_ddp
        self.monitor = monitor
        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"log.txt")

        if self.use_ddp:
            self._setup_ddp()
        else:
            self._setup_device()

        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

        self.model.to(self.device) 


    def _setup_device(self) -> None:
        """
        Setup the device for training. Use GPU if available, otherwise use CPU.
        """
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.main_process = True


    def _setup_ddp(self) -> None:
        """
        Setup Distributed Data Parallel (DDP) for training.
        """
        self.ddp_rank = int(os.environ['RANK'])
        self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        self.ddp_world_size = int(os.environ['WORLD_SIZE'])

        self.device = f'cuda:{self.ddp_local_rank}'
        torch.cuda.set_device(self.device)

        self.main_process = self.ddp_rank == 0


    def train(self) -> None:
        """
        Train the model.
        """
        pass


    @torch.no_grad()
    def validate(self) -> None:
        """
        Validate the model on the validation dataset.
        """
        pass


    @torch.no_grad()
    def evaluate(self) -> None:
        """
        Evaluate the model. 
        """
        pass


    def get_lr(self, iter: int) -> float:
        """
        Calculate the learning rate at a given iteration.

        :param iter: The current iteration.
        :return: The calculated learning rate.
        """
        if iter < self.warmup_iters:
            return self.max_lr * (iter + 1) / self.warmup_iters
        
        if iter > self.max_iters:
            return self.min_lr
        
        decay_ratio = (iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)

        assert 0 <= decay_ratio <= 1, "Decay ratio out of range."

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return self.min_lr + coeff * (self.max_lr - self.min_lr)


    def get_optimizer(self):
        pass
