import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models._modules import *
from typing import Dict, Any


class GPT(nn.Module):
    """
    GPT Model

    This module applies a GPT-like architecture with token and positional embeddings,
    followed by a series of transformer blocks and a final normalization and output layer.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        Initializes the GPT module.

        :param cfg: Configuration dictionary containing the following keys:
            - "vocab_size": int, the size of the vocabulary.
            - "emb_dim": int, the embedding dimension.
            - "context_length": int, the length of the input sequences.
            - "drop_rate": float, the dropout rate.
            - "n_layers": int, the number of transformer layers.
            - "n_heads": int, the number of attention heads.
            - "qkv_bias": bool, whether to include bias in the query, key, value projections.
        :type cfg: Dict[str, Any]
        """
        super().__init__()
        self.n_layers = cfg['n_layers']
        self.gpt = nn.ModuleDict({
            "t_embedding": nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]),
            "p_embedding": nn.Embedding(cfg["context_length"], cfg["emb_dim"]),
            "dropout": nn.Dropout(cfg["drop_rate"]),
            "transformers": nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]),
            "final_ln": LayerNorm(cfg["emb_dim"]),
            "ln_out": nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        })

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
    

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Applies the GPT model to the input tensor.

        :param in_idx: The input tensor containing token indices.
        :type in_idx: torch.Tensor
        :return: The tensor containing the logits for each token in the vocabulary.
        :rtype: torch.Tensor
        """
        _, n_tokens = in_idx.shape

        token_embeddings = self.gpt["t_embedding"](in_idx)
        positional_embeddings = self.gpt["p_embedding"](torch.arange(n_tokens, device=in_idx.device))
        x = token_embeddings + positional_embeddings
        x = self.gpt["dropout"](x)
        x = self.gpt["transformers"](x)
        x = self.gpt["final_ln"](x)
        logits = self.gpt["ln_out"](x)

        return logits