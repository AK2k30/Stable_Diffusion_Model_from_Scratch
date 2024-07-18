import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

"""
    Implements a cross-attention mechanism, where the query tensor is derived from one input tensor (x) and the key and value tensors are derived from another input tensor (y).

    This module takes two input tensors `x` and `y` and computes the cross-attention output. It first projects the input tensors `x` and `y` into query, key, and value tensors using linear layers. Then, it computes the attention weights by taking the dot product of the query and key tensors, scales the weights by the square root of the head dimension, and applies a softmax. Finally, it computes the output by taking the weighted sum of the value tensor.

    The attention mechanism supports both regular attention and causal (masked) attention, where future positions are masked out to prevent information leakage.
""" 
class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(n_vocab, n_embd)
        self.position_embeddings = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        x = self.token_embeddings(tokens)
        x += self.position_embeddings

        return x

"""
    Implements a single layer of the CLIP transformer model.

    This layer consists of a self-attention mechanism followed by a feedforward neural network. The self-attention mechanism computes the attention weights between the input tokens, and the feedforward network applies a non-linear transformation to the attended input.

    The layer also includes layer normalization before and after the attention and feedforward computations, as well as residual connections around each of these components.
"""
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residue

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear_2(x)

        x += residue

        return x

"""
Implements the CLIP (Contrastive Language-Image Pre-training) model.

The CLIP model is a neural network that learns visual representations from natural language supervision. It consists of an image encoder and a text encoder, which are trained jointly to predict the correct pairings of images and text.

This class represents the overall CLIP model, which includes an embedding layer, a stack of CLIP transformer layers, and a final layer normalization.
"""
class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long) 

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output


