"""GPT-2-style transformer in PyTorch for Sudoku solving traces."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    vocab_size: int = 731
    max_seq_len: int = 82
    dtype: str = "float32"

    @property
    def torch_dtype(self):
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        cfg = self.config
        B, T, C = x.shape
        head_dim = C // cfg.n_heads

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)  # each (B, T, C)

        q = q.reshape(B, T, cfg.n_heads, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.reshape(B, T, cfg.n_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, T, cfg.n_heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask[None, None, :, :], float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.fc = nn.Linear(config.d_model, config.d_ff)
        self.proj = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        # Pre-norm: LN -> attention -> residual
        x = x + self.attn(self.ln1(x))
        # Pre-norm: LN -> FFN -> residual
        h = self.ln2(x)
        h = F.gelu(self.fc(h))
        h = self.proj(h)
        x = x + h
        return x


class GPT2Model(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, return_intermediates=False):
        B, T = tokens.shape
        device = tokens.device

        tok_emb = self.token_emb(tokens)
        pos_emb = self.pos_emb(torch.arange(T, device=device).unsqueeze(0))
        x = tok_emb + pos_emb

        intermediates = []
        for block in self.blocks:
            x = block(x)
            if return_intermediates:
                intermediates.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_intermediates:
            return logits, torch.stack(intermediates)
        return logits
