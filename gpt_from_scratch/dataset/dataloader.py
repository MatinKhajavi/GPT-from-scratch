import os
import numpy as np
import torch
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class DataLoaderConfig:
    """
    Configuration settings for the DataLoader.

    :param n_batches: Number of batches.
    :type n_batches: int
    :param n_tokens: Number of tokens per batch.
    :type n_tokens: int
    :param data_root: Directory name where data shards are stored.
    :type data_root: str
    :param process_rank: Rank of the current process (for parallel processing).
    :type process_rank: int
    :param n_processes: Total number of processes involved.
    :type n_processes: int
    :param main_process: Flag to indicate if this is the main process.
    :type main_process: bool
    :param split: Type of data split ('train' or 'val').
    :type split: str    
    """

    n_batches: int
    n_tokens: int
    data_root: str
    process_rank: int
    n_processes: int
    main_process: bool
    split: str


class DataLoader:
    """
    DataLoader class for loading and processing batches of tokenized data.

    This class handles loading data from shards, which are split by training or validation sets, and
    ensures data is distributed evenly across different processes in a parallel computing environment.
    """

    def __init__(self, cfg: DataLoaderConfig) -> None:
        """
        Initializes the DataLoader object.

        :raises AssertionError: If no shards are found for the specified split.
        """

        self.cfg = cfg
        assert cfg.split in {'train', 'val'}, "Invalid split. It must be 'train' or 'val'."


        shards = sorted([os.path.join(cfg.data_root, s) for s in os.listdir(cfg.data_root) if cfg.split in s])

        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {self.cfg.split}"

        if self.cfg.main_process:
            print(f"found {len(shards)} shards for split {self.cfg.split}")
        
        self.reset()

    def reset(self) -> None:
        """
        Resets the data loader to the beginning of the first shard.
        """
        self.current_shard: int = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position: int = self.cfg.n_batches * self.cfg.n_tokens * self.cfg.process_rank

    def load_tokens(self, name: str) -> torch.Tensor:
        """
        Loads tokens from a file.

        :param name: The file path to load data from.
        :type name: str
        :return: A tensor of tokens.
        :rtype: torch.Tensor
        """
        npt = np.load(name)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the next batch of tokens.

        :return: A tuple containing the input batch 'x' and the target batch 'y'.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        buf = self.tokens[self.current_position : self.current_position + self.cfg.n_batches * self.cfg.n_tokens + 1]
        x = buf[:-1].view(self.cfg.n_batches, self.cfg.n_tokens) 
        y = buf[1:].view(self.cfg.n_batches, self.cfg.n_tokens) 

        self.current_position += self.cfg.n_batches * self.cfg.n_tokens * self.cfg.n_processes

        if self.current_position + (self.cfg.n_batches * self.cfg.n_tokens * self.cfg.n_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = self.cfg.n_batches * self.cfg.n_tokens * self.cfg.process_rank
        
        return x, y

