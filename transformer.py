"""GPT-2-style transformer in Flax for Sudoku solving traces."""

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass
class TransformerConfig:
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    vocab_size: int = 731
    max_seq_len: int = 82 # Since there are only 81 cells, we can use a shorter sequence length, even if we'll allow to backtrack eventually


class CausalSelfAttention(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        B, T, C = x.shape
        head_dim = C // cfg.n_heads
        assert C % cfg.n_heads == 0

        qkv = nn.Dense(3 * C, name="qkv")(x)  # (B, T, 3C)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)

        scale = head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, nh, T, T)

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn = jnp.where(mask[None, None, :, :], attn, -1e9)
        attn = nn.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)  # (B, T, C)
        out = nn.Dense(C, name="proj")(out)
        return out


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        # Pre-norm: LN -> attention -> residual
        h = nn.LayerNorm()(x)
        h = CausalSelfAttention(cfg)(h)
        x = x + h
        # Pre-norm: LN -> FFN -> residual
        h = nn.LayerNorm()(x)
        h = nn.Dense(cfg.d_ff)(h)
        h = nn.gelu(h)
        h = nn.Dense(cfg.d_model)(h)
        x = x + h
        return x


class GPT2Model(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, tokens, return_intermediates=False):
        cfg = self.config
        B, T = tokens.shape

        tok_emb = nn.Embed(cfg.vocab_size, cfg.d_model, name="token_emb")(tokens)
        pos_emb = nn.Embed(cfg.max_seq_len, cfg.d_model, name="pos_emb")(
            jnp.arange(T)[None, :]  # (1, T)
        )
        x = tok_emb + pos_emb  # (B, T, d_model)

        intermediates = []
        for i in range(cfg.n_layers):
            x = TransformerBlock(cfg, name=f"block_{i}")(x)
            if return_intermediates:
                intermediates.append(x)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(cfg.vocab_size, name="lm_head")(x)  # (B, T, vocab_size)

        if return_intermediates:
            # Stack to (n_layers, B, T, d_model)
            return logits, jnp.stack(intermediates)
        return logits
