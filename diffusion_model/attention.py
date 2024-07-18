import torch
from torch import nn
from torch.nn import functional as F
import math

"""
Implements a self-attention mechanism, where the query, key, and value tensors are all derived from the same input tensor.

This module takes an input tensor `x` and computes the self-attention output. It first projects the input tensor into query, key, and value tensors using linear layers. Then, it computes the attention weights by taking the dot product of the query and key tensors, scales the weights by the square root of the head dimension, and applies a softmax. Finally, it computes the output by taking the weighted sum of the value tensor.

The attention mechanism supports both regular attention and causal (masked) attention, where future positions are masked out to prevent information leakage.
"""
class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunks(3, dim=1)

        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.sotmax(weight, dim=1)

        output = weight @ v

        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output

"""
    Implements a cross-attention mechanism, where the query tensor is derived from one input tensor (x) and the key and value tensors are derived from another input tensor (y).

    This module takes two input tensors `x` and `y` and computes the cross-attention output. It first projects the input tensors `x` and `y` into query, key, and value tensors using linear layers. Then, it computes the attention weights by taking the dot product of the query and key tensors, scales the weights by the square root of the head dimension, and applies a softmax. Finally, it computes the output by taking the weighted sum of the value tensor.

    The attention mechanism supports both regular attention and causal (masked) attention, where future positions are masked out to prevent information leakage.
"""
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output




