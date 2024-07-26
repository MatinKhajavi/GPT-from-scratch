import os
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class Trainer:

    def __init__(self,
                model,
                train_loader,
                val_loader=None,
                max_iters=19073,
                max_lr = 6e-4,
                min_lr=6e-3,
                monitor=True,
                log_dir="log") -> None:
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_iters = max_iters
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.use_ddp = int(os.environ.get('RANK', -1)) != -1
        self.monitor = monitor
        self.log_dir = log_dir

        if self.use_ddp:
            self._setup_ddp()
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1

            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            
            self.main_process = True
        
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"


    def _setup_ddp(self):
        self.ddp_rank = int(os.environ['RANK'])
        self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        self.ddp_world_size = int(os.environ['WORLD_SIZE'])

        self.device = f'cuda:{self.ddp_local_rank}'
        torch.cuda.set_device(self.device)

        self.main_process = self.ddp_rank == 0


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