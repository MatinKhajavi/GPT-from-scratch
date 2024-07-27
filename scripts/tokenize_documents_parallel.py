import os
import argparse
import multiprocessing as mp
from datasets import load_dataset
from src.dataset.tokenizer import Tokenizer
from src.dataset.shardmanager import ShardManager

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenizes documents using multiprocessing.")
    parser.add_argument("--local_dir", type=str, default="edu_fineweb10B", help="Local directory to store data.")
    parser.add_argument("--remote_name", type=str, default="sample-10BT", help="Remote dataset name.")
    parser.add_argument("--shard_size", type=int, default=int(1e8), help="Number of tokens per shard.")
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb-edu", help="Path to the dataset on Hugging Face.")
    parser.add_argument("--tokenizer_model", type=str, default="gpt2", help="Tokenizer model to use.")
    return parser.parse_args()

def main():
    args = parse_args()

    current_script_path = os.path.dirname(__file__)
    project_root = os.path.dirname(current_script_path)
    data_dir = os.path.join(project_root, 'data')
    DATA_CACHE_DIR = os.path.join(data_dir, args.local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    fw = load_dataset(args.dataset_path, name=args.remote_name, split="train")

    tokenizer = Tokenizer(args.tokenizer_model)
    shard_manager = ShardManager(DATA_CACHE_DIR, args.shard_size)

    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenizer.tokenize_doc, fw, chunksize=16):
            shard_manager.add_tokens(tokens)

    shard_manager.finalize()

if __name__ == "__main__":
    main()
