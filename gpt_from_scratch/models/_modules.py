import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from gpt_from_scratch.models.utils import GPTConfig


class MHAttention(nn.Module):
    """
    Implements a multihead attention with PyTorch's scaled dot product attention.

    :param d_in: Input embedding size.
    :type d_in: int
    :param d_out: Output embedding size. Must be divisible by n_heads.
    :type d_out: int
    :param n_heads: Number of attention heads.
    :type n_heads: int
    :param dropout: Dropout probability (default: 0.0).
    :type dropout: float, optional
    :param qkv_bias: Whether to add a bias term to the QKV transformations (default: False).
    :type qkv_bias: bool, optional

    The model expects input tensors of shape (batch_size, number_of_tokens, d_in).
    The output tensors are of shape (batch_size, number_of_tokens, d_out).
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int, dropout: float = 0.1, qkv_bias: bool = False) -> None:
        super().__init__()

        assert d_out % n_heads == 0, "Output embedding size (d_out) must be divisible by n_heads"

        self.d_out = d_out
        self.d_head = d_out // n_heads
        self.n_heads = n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.proj.SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention mechanism.

        :param x: Input tensor of shape (batch_size, number_of_tokens, d_in).
        :type x: torch.Tensor
        :returns: Output tensor of shape (batch_size, number_of_tokens, d_out).
        :rtype: torch.Tensor
        """
        n_batches, n_tokens, _ = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.split(self.d_out, dim=2)
        key = k.view(n_batches, n_tokens, self.n_heads, self.d_head).transpose(1, 2) 
        query = q.view(n_batches, n_tokens, self.n_heads, self.d_head).transpose(1, 2) 
        value = v.view(n_batches, n_tokens, self.n_heads, self.d_head).transpose(1, 2)

        use_dropout = self.dropout if self.training else 0.0

        context_vec = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=use_dropout)
        context_vec = context_vec.transpose(1, 2).contiguous().view(n_batches, n_tokens, self.d_out)

        context_vec = self.proj(context_vec)
        return context_vec


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    """

    def __init__(self) -> None:
        """
        Initializes the GELU activation module.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the GELU activation function to the input tensor.

        :param x: The input tensor to which the GELU activation will be applied.
        :type x: torch.Tensor
        :return: The tensor after applying the GELU activation function.
        :rtype: torch.Tensor
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class LayerNorm(nn.Module):
    """
    Layer Normalization module.

    This module applies layer normalization to the input tensor.
    """

    def __init__(self, emb_dim: int) -> None:
        """
        Initializes the LayerNorm module.

        :param emb_dim: The dimension of the input embedding.
        :type emb_dim: int
        """
        super().__init__()
        self.eps: float = 1e-5
        self.scale: nn.Parameter = nn.Parameter(torch.ones(emb_dim))
        self.shift: nn.Parameter = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies layer normalization to the input tensor.

        :param x: The input tensor to which the layer normalization will be applied.
        :type x: torch.Tensor
        :return: The tensor after applying layer normalization.
        :rtype: torch.Tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    """
    FeedForward neural network module.
    """

    def __init__(self, emb_dim: int) -> None:
        """
        Initializes the FeedForward module.

        :param emb_dim: The embedding dimension.
        :type emb_dim: int
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            # GELU()
            nn.GELU(approximate='tanh')
        )
        self.final_l = nn.Linear(4 * emb_dim, emb_dim)
        self.final_l.SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feedforward neural network to the input tensor.

        :param x: The input tensor to which the feedforward network will be applied.
        :type x: torch.Tensor
        :return: The tensor after applying the feedforward network.
        :rtype: torch.Tensor
        """
        x = self.layers(x)
        return self.final_l(x)
    

class TransformerBlock(nn.Module):
    """
    Transformer block module.

    This module applies a multi-head attention mechanism followed by a feed-forward neural network,
    with layer normalization and dropout applied at each step.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        """
        Initializes the TransformerBlock module.

        :param cfg: Configuration settings containing the following keys:
            - "emb_dim": int, the embedding dimension.
            - "context_length": int, the length of the input sequences.
            - "n_heads": int, the number of attention heads.
            - "drop_rate": float, the dropout rate.
            - "qkv_bias": bool, whether to include bias in the query, key, value projections.
        :type cfg: GPTConfig
        """
        super().__init__()
        self.mha = MHAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            n_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        self.ff = FeedForward(cfg.emb_dim)
        self.norm1 = nn.LayerNorm(cfg.emb_dim)
        self.norm2 = nn.LayerNorm(cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformer block to the input tensor.

        :param x: The input tensor to which the transformer block will be applied.
        :type x: torch.Tensor
        :return: The tensor after applying the transformer block.
        :rtype: torch.Tensor
        """
        shortcut = x
        x = self.norm1(x)
        x = self.mha(x) 
        x = self.dropout(x)
        x = x + shortcut  

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut  

        return x
    
