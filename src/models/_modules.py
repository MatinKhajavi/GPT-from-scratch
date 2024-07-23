import torch
import torch.nn as nn
import torch.nn.functional as F

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
