"""
This script includes code adapted from the following source:
https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py

Original code by Andrej Karpathy, licensed under the MIT License.
"""


import os
import time
import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group
import torch.distributed as dist
import math
from typing import Optional, Any, Union
from src.dataset import DataLoader
import inspect
from torch.optim import Optimizer
from src.metrics import hellaswag_evaluation
from src.dataset import tokenize_str, decode_tokens

class Trainer:
    """
    Trainer class for managing the training process.

    :param model: The model to be trained.
    :param train_loader: The data loader for the training dataset.
    :param val_loader: The data loader for the validation dataset.
    :param raw_model: The original neural network model, unwrapped from DDP if applicable.
    :param n_epochs: The number of epochs.
    :param warmup_iters: Number of iterations for learning rate warmup.
    :param max_iters: Total number of iterations for training.
    :param grad_accum_iters: The number of iterations for gradient accumulation.
    :param metrics: A list of metrics to evaluate the model.
    :param max_lr: Maximum learning rate.
    :param min_lr: Minimum learning rate.
    :param use_ddp: Flag to setup DDP or not.
    :param device: The device to run the training on (e.g., "cpu", "cuda").
    :param ddp_rank: The rank of the process in DDP training.
    :param ddp_local_rank: The local rank of the process in DDP training.
    :param ddp_world_size: The total number of processes in DDP training.
    :param main_process: Flag to indicate if the current process is the main process.
    :param monitor: Flag to monitor the training process.
    :param torch_matmul_percision: Set the precision of floating-point matrix multiplications.
    :param log_dir: Directory to store logs.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 raw_model: Optional[torch.nn.Module] = None,
                 n_epochs: int = 1,
                 warmup_iters: int = 715,
                 max_iters: int = 19073,
                 grad_accum_iters: int = 1,
                 metrics: list[str] = ["Hellaswag"],
                 max_lr: float = 6e-4,
                 min_lr: float = 6e-5,
                 use_ddp: bool = False,
                 device: str = "cpu",
                 ddp_rank: int = 0,
                 ddp_local_rank: int = 0,
                 ddp_world_size: int = 1,
                 main_process: bool = True,
                 monitor: bool = True,
                 torch_matmul_percision: str = "high",
                 log_dir: str = "log") -> None:
        
        self.model = model
        self.raw_model = raw_model
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.grad_accum_iters = grad_accum_iters
        self.metrics = metrics
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.use_ddp = use_ddp
        self.device = device
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.main_process = main_process
        self.monitor = monitor

        torch.set_float32_matmul_precision(torch_matmul_percision)
        
        if use_ddp:
            torch.cuda.set_device(self.device)

        self.log_dir = log_dir

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"log.txt")
        with open(self.log_file, "w") as _:
            pass

        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"


    def train(self) -> None:
        """
        Train the model.
        """
        
        self.optimizer = self._get_optimizer(weight_decay=0.1, learning_rate=6e-4)

        for epoch in range(self.n_epochs):
            self._on_epoch_begin()
            for iter in range(self.max_iters):
                t0 = time.time()

                self.model.train()
                self.optimizer.zero_grad()
                accumulated_loss = 0.0

                for micro_iter in range(self.grad_accum_iters):
                    x, y = self.train_loader.next_batch()
                    x, y = x.to(self.device), y.to(self.device)

                    if self.use_ddp:
                        self.model.require_backward_grad_sync = (micro_iter == self.grad_accum_iters - 1)
                    with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                        logits = self.model(x)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                    loss = loss / self.grad_accum_iters
                    accumulated_loss += loss.detach()
                    loss.backward()
                
                if self.use_ddp:
                    dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG)
                
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                lr = self.adjust_optimizer_lr(iter)
                
                self.optimizer.step()

                is_last_step = (iter == self.max_iters - 1)

                if iter % 250 == 0 or is_last_step:
                    self.validate(epoch, iter, is_last_step)
                    self.evaluate(epoch, iter)

                    
                    idx = tokenize_str("Once upon a time")
                    idx = idx.to(self.device)
                    generated = self.raw_model.generate(idx, 25)
                    gen_str = decode_tokens(generated)
                    print(f"GPU {self.ddp_rank}: {gen_str}")
                

                if self.device_type == "cuda":
                    torch.cuda.synchronize() 
                
                dt = time.time() - t0 

                tokens_processed = self.train_loader.cfg.n_batches * self.train_loader.cfg.n_tokens * self.grad_accum_iters * self.ddp_world_size
                tokens_per_sec = tokens_processed / dt

                if self.main_process:
                    if self.monitor:
                        print(f"Epoch {epoch} | iter {iter:5d} | loss: {accumulated_loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
                    with open(self.log_file, "a") as f:
                        f.write(f"Epoch {epoch} | iter {iter} | train loss: {accumulated_loss.item():.6f}\n")
        
        if self.use_ddp:
            destroy_process_group()

    
    def adjust_optimizer_lr(self, iter: int) -> float:
        """
        Adjust the learning rate of the optimizer based on the current iteration.

        :param iter: The current iteration.
        :type iter: int
        :return: The adjusted learning rate.
        :rtype: float
        """
        lr = self._get_lr(iter)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

    def _on_epoch_begin(self) -> None:
        """
        Perform operations at the beginning of each epoch.
        """
        self.train_loader.reset()


    @torch.no_grad()
    def validate(self, epoch, iter, is_last_iter) -> None:
        """
        Validate the model on the validation dataset.

        :param epoch: The current epoch.
        :type epoch: int
        :param iter: The current iteration.
        :type iter: int
        :param is_last_iter: A flag indicating whether this is the last iteration.
        :type is_last_iter: bool
        """
        
        self.model.eval()
        self.val_loader.reset()

        val_loss_accum = 0.0
        val_loss_iters = 20
        for _ in range(val_loss_iters):
            x, y = self.val_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / val_loss_iters
            val_loss_accum += loss.detach()
        if self.use_ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.main_process:
            if self.monitor:
                print(f"Validation loss: {val_loss_accum.item():.3f}")
            with open(self.log_file, "a") as f:
                f.write(f"Epoch {epoch} | iter {iter} | validation loss: {val_loss_accum.item():.3f}\n")
            if iter > 0 and (iter % 5000 == 0 or is_last_iter):
                checkpoint_path = os.path.join(self.log_dir, f"model_epoch{epoch}_iter{iter:05d}.pt")
                checkpoint = {
                    'model': self.raw_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'config': self.raw_model.cfg,
                    'epoch': epoch,
                    'iter': iter,
                    'val_loss': val_loss_accum.item()
                }

                torch.save(checkpoint, checkpoint_path)


    @torch.no_grad()
    def evaluate(self, epoch, iter) -> None:
        """
        Evaluate the model on different metrics.

        :param epoch: The current epoch.
        :type epoch: int
        :param iter: The current iteration.
        :type iter: int
        """
        self.model.eval() 
        if "Hellaswag" in self.metrics:
            num_correct_norm, num_total = hellaswag_evaluation(self.model, self.ddp_world_size,
                                                               self.ddp_rank, self.device, self.device_type)
            
            if self.use_ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=self.device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=self.device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            
            acc_norm = num_correct_norm / num_total
            if self.main_process:
                if self.monitor:
                    print(f"Hellaswag accuracy: {acc_norm:.3f}")
                with open(self.log_file, "a") as f:
                    f.write(f"Epoch {epoch} | iter {iter} | Hellaswag accuracy: {acc_norm:.3f}\n")


    def _get_lr(self, iter: int) -> float:
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


    def _get_optimizer(self, weight_decay: float, learning_rate: float) -> Optimizer:
        """
        Get the optimizer for the training process.

        :param weight_decay: The weight decay to be applied to the optimizer.
        :param learning_rate: The learning rate for the optimizer.
        :return: The created AdamW optimizer.
        """
        # Start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.raw_model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        if self.main_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device_type == "cuda"

        if self.main_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=learning_rate, 
            betas=(0.9, 0.95), 
            eps=1e-8, 
            fused=use_fused
        )

        return optimizer