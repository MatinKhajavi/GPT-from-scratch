import os
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from src.models import GPT, GPTConfig
from src.trainers import Trainer
from src.dataset import DataLoader, DataLoaderConfig
from torch.distributed import init_process_group


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training Script")
    parser.add_argument("--n_batches", type=int, default=16, help="Number of batches")
    parser.add_argument("--n_tokens", type=int, default=1024, help="Number of tokens")
    parser.add_argument("--data_root", type=str, default="data/edu_fineweb10B", help="Data root directory")
    parser.add_argument("--vocab_size", type=int, default=50304, help="Vocabulary size")
    parser.add_argument("--emb_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--qkv_bias", action='store_true', default=False, help="Use bias in QKV projection")
    parser.add_argument("--monitor", action='store_true', default=True, help="Enable monitoring")
    parser.add_argument("--torch_matmul_precision", type=str, default="high", help="Torch matmul precision")
    parser.add_argument("--log_dir", type=str, default="log", help="Log directory")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--warmup_iters", type=int, default=715, help="Number of warmup iterations")
    parser.add_argument("--max_iters", type=int, default=19073, help="Maximum number of iterations")
    parser.add_argument("--total_batch_size", type=int, default=2**19, help="Total batch size")
    parser.add_argument("--metrics", type=str, nargs='+', default=["Hellaswag"], help="Metrics to evaluate")
    
    return parser.parse_args()


def main():
    args = parse_args()

    n_batches = args.n_batches
    n_tokens = args.n_tokens

    use_ddp = int(os.environ.get('RANK', -1)) != -1
    if use_ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        main_process = ddp_rank == 0

    else:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        main_process = True

    train_data_cfg = DataLoaderConfig(n_batches=n_batches, n_tokens=n_tokens, data_root=args.data_root,
                                    n_processes=ddp_world_size, process_rank=ddp_rank, main_process=main_process, split="train")
    train_loader = DataLoader(train_data_cfg)

    val_data_cfg = DataLoaderConfig(n_batches=n_batches, n_tokens=n_tokens, data_root=args.data_root,
                                    n_processes=ddp_world_size, process_rank=ddp_rank, main_process=main_process, split="val")
    val_loader = DataLoader(val_data_cfg)

    model_cfg = GPTConfig(
        vocab_size=args.vocab_size,
        emb_dim=args.emb_dim,
        context_length=args.context_length,
        drop_rate=args.drop_rate,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        qkv_bias=args.qkv_bias
    )
    model = GPT(model_cfg)
    model.to(device)
    
    if use_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if use_ddp else model

    grad_accum_iters = args.total_batch_size // (n_batches * n_tokens * ddp_world_size)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, raw_model=raw_model, use_ddp=use_ddp,
                    device=device, ddp_rank=ddp_rank, ddp_local_rank=ddp_local_rank, ddp_world_size=ddp_world_size,
                    main_process=main_process, monitor=args.monitor, torch_matmul_percision=args.torch_matmul_precision,
                    log_dir=args.log_dir, n_epochs=args.n_epochs, warmup_iters=args.warmup_iters, max_iters=args.max_iters,
                    grad_accum_iters=grad_accum_iters, metrics=args.metrics)
    
    trainer.train()


if __name__ == "__main__":
    main()
