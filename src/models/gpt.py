import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models._modules import *
from typing import Dict, Any
from typing import Optional
from src.models.utils import GPTConfig

class GPT(nn.Module):
    """
    GPT Model

    This module applies a GPT architecture with token and positional embeddings,
    followed by a series of transformer blocks and a final normalization and output layer.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """
        Initializes the GPT module.

        :param cfg: Configuration settings containing the following keys:
            - "vocab_size": int, the size of the vocabulary.
            - "emb_dim": int, the embedding dimension.
            - "context_length": int, the length of the input sequences.
            - "drop_rate": float, the dropout rate.
            - "n_layers": int, the number of transformer layers.
            - "n_heads": int, the number of attention heads.
            - "qkv_bias": bool, whether to include bias in the query, key, value projections.
        :type cfg: GPTConfig
        """
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_layers
        self.gpt = nn.ModuleDict({
            "t_embedding": nn.Embedding(cfg.vocab_size, cfg.emb_dim),
            "p_embedding": nn.Embedding(cfg.context_length, cfg.emb_dim),
            "dropout": nn.Dropout(cfg.drop_rate),
            "transformers": nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            "final_ln": nn.LayerNorm(cfg.emb_dim),
            "ln_out": nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
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
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = 50) -> torch.Tensor:
        """
        Generate new tokens based on the given input indices.

        This function generates new tokens for the input sequence up to a specified
        number of new tokens. The process involves adjusting the logits using the
        provided temperature and optionally applying top-k sampling.

        :param idx: A tensor of shape (batch_size, sequence_length) containing the input indices.
        :type idx: torch.Tensor
        :param max_new_tokens: The maximum number of new tokens to generate.
        :type max_new_tokens: int
        :param temperature: The temperature value for scaling the logits. Default is 1.0.
        :type temperature: float
        :param top_k: The number of top logits to consider for sampling. Default is 50.
        :type top_k: Optional[int]
        :return: A tensor containing the input indices concatenated with the generated new tokens.
        :rtype: Tensor
        """

        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            assert temperature >= 0.0

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)  
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
            


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