import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class MHAttention(nn.Module):
    """
    Implements a multihead attention with PyTorch's scaled dot product attention.

    :param d_in: Input embedding size.
    :type d_in: int
    :param d_out: Output embedding size. Must be divisible by n_heads.
    :type d_out: int
    :param n_heads: Number of attention heads.
    :type n_heads: int
    :param number_of_tokens: Number of tokens in the input.
    :type number_of_tokens: int
    :param dropout: Dropout probability (default: 0.0).
    :type dropout: float, optional
    :param qkv_bias: Whether to add a bias term to the QKV transformations (default: False).
    :type qkv_bias: bool, optional

    The model expects input tensors of shape (batch_size, number_of_tokens, d_in).
    The output tensors are of shape (batch_size, number_of_tokens, d_out).
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int, number_of_tokens: int, dropout: float = 0.0, qkv_bias: bool = False) -> None:
        super().__init__()

        assert d_out % n_heads == 0, "Output embedding size (d_out) must be divisible by n_heads"

        self.d_out = d_out
        self.d_head = d_out // n_heads
        self.n_heads = n_heads
        self.number_of_tokens = number_of_tokens
        self.dropout = dropout

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention mechanism.

        :param x: Input tensor of shape (batch_size, number_of_tokens, d_in).
        :type x: torch.Tensor
        :returns: Output tensor of shape (batch_size, number_of_tokens, d_out).
        :rtype: torch.Tensor
        """
        n_batch, n_tokens, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(n_batch, n_tokens, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.chunk(3, 0)

        use_dropout = self.dropout if self.training else 0.0

        context_vec = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=use_dropout)
        context_vec = context_vec.transpose(1, 2).contiguous().view(n_batch, n_tokens, self.d_out)

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

    def __init__(self, cfg: Dict[str, int]) -> None:
        """
        Initializes the FeedForward module.

        :param cfg: Configuration dictionary containing the embedding dimension.
        :type cfg: Dict[str, int]
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feedforward neural network to the input tensor.

        :param x: The input tensor to which the feedforward network will be applied.
        :type x: torch.Tensor
        :return: The tensor after applying the feedforward network.
        :rtype: torch.Tensor
        """
        return self.layers(x)