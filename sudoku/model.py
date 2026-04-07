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
    vocab_size: int = 731   # standard vocab; BT format uses 734 — always set explicitly
    max_seq_len: int = 82   # standard format; BT format uses longer sequences — always set explicitly
    use_pos_emb: bool = True
    dtype: str = "float32"  # "float32", "bfloat16", or "float16"

    @property
    def jax_dtype(self):
        return {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}[self.dtype]


class CausalSelfAttention(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, cache=None, cache_index=None, attn_mask=None):
        cfg = self.config
        B, T, C = x.shape
        head_dim = C // cfg.n_heads
        assert C % cfg.n_heads == 0

        dtype = cfg.jax_dtype
        qkv = nn.Dense(3 * C, dtype=dtype, name="qkv")(x)  # (B, T, 3C)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            max_len = cfg.max_seq_len
            # Write new K, V into cache using dynamic_update_slice
            k_cache = jax.lax.dynamic_update_slice(cache["k"], k, (0, 0, cache_index, 0))
            v_cache = jax.lax.dynamic_update_slice(cache["v"], v, (0, 0, cache_index, 0))
            updated_cache = {"k": k_cache, "v": v_cache}

            # Attend to full cache (unused slots are zeros, masked out below)
            scale = head_dim ** -0.5
            attn = (q @ k_cache.transpose(0, 1, 3, 2)) * scale  # (B, nh, T, max_len)

            # Causal mask: query at absolute pos i can attend to key at pos j if j <= i
            q_pos = jnp.arange(T) + cache_index          # (T,)
            k_pos = jnp.arange(max_len)                   # (max_len,)
            mask = k_pos[None, :] <= q_pos[:, None]        # (T, max_len)
            attn = jnp.where(mask[None, None, :, :], attn, jnp.finfo(dtype).min)

            attn = nn.softmax(attn, axis=-1)
            out = (attn @ v_cache).transpose(0, 2, 1, 3).reshape(B, T, C)
            out = nn.Dense(C, dtype=dtype, name="proj")(out)
            return out, updated_cache

        scale = head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, nh, T, T)

        # Causal mask (or custom packed-sequence mask)
        if attn_mask is not None:
            attn = jnp.where(attn_mask, attn, jnp.finfo(dtype).min)
        else:
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            attn = jnp.where(mask[None, None, :, :], attn, jnp.finfo(dtype).min)
        attn = nn.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)  # (B, T, C)
        out = nn.Dense(C, dtype=dtype, name="proj")(out)
        return out


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, cache=None, cache_index=None, attn_mask=None, return_intermediates=False):
        cfg = self.config
        dtype = cfg.jax_dtype
        # Pre-norm: LN -> attention -> residual
        h = nn.LayerNorm(dtype=dtype)(x)
        if cache is not None:
            h, updated_cache = CausalSelfAttention(cfg)(h, cache=cache, cache_index=cache_index)
        else:
            h = CausalSelfAttention(cfg)(h, attn_mask=attn_mask)
            updated_cache = None
        x = x + h
        post_attn = x
        # Pre-norm: LN -> FFN -> residual
        h = nn.LayerNorm(dtype=dtype)(x)
        h = nn.Dense(cfg.d_ff, dtype=dtype)(h)
        h = nn.gelu(h)
        h = nn.Dense(cfg.d_model, dtype=dtype)(h)
        x = x + h
        if return_intermediates:
            layer_acts = {"post_attn": post_attn, "post_mlp": x}
            if updated_cache is not None:
                return x, updated_cache, layer_acts
            return x, layer_acts
        if updated_cache is not None:
            return x, updated_cache
        return x


class GPT2Model(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, tokens, cache=None, cache_index=None, return_intermediates=False,
                 attn_mask=None, positions=None):
        cfg = self.config
        B, T = tokens.shape

        dtype = cfg.jax_dtype
        tok_emb = nn.Embed(cfg.vocab_size, cfg.d_model, dtype=dtype, name="token_emb")(tokens)

        if cfg.use_pos_emb:
            if positions is not None:
                pos_ids = positions  # (B, T) — per-document local positions for packing
            elif cache_index is not None:
                pos_ids = jax.lax.dynamic_slice(jnp.arange(cfg.max_seq_len), (cache_index,), (T,))[None, :]
            else:
                pos_ids = jnp.arange(T)[None, :]
            pos_emb = nn.Embed(cfg.max_seq_len, cfg.d_model, dtype=dtype, name="pos_emb")(pos_ids)
            x = tok_emb + pos_emb  # (B, T, d_model)
        else:
            x = tok_emb

        updated_caches = []
        intermediates: dict[str, jnp.ndarray] = {}
        for i in range(cfg.n_layers):
            layer_cache = cache[i] if cache is not None else None
            if layer_cache is not None:
                x, new_cache = TransformerBlock(cfg, name=f"block_{i}")(x, cache=layer_cache, cache_index=cache_index)
                updated_caches.append(new_cache)
            else:
                if return_intermediates:
                    x, layer_acts = TransformerBlock(cfg, name=f"block_{i}")(x, attn_mask=attn_mask, return_intermediates=True)
                    for k, v in layer_acts.items():
                        intermediates[f"layer_{i}_{k}"] = v
                else:
                    x = TransformerBlock(cfg, name=f"block_{i}")(x, attn_mask=attn_mask)

        x = nn.LayerNorm(dtype=dtype)(x)
        logits = nn.Dense(cfg.vocab_size, dtype=dtype, name="lm_head")(x)  # (B, T, vocab_size)

        if cache is not None:
            return logits, updated_caches
        if return_intermediates:
            # intermediates: dict mapping "layer_{i}_{descriptor}" -> (batch, seq_len, d_model)
            return logits, intermediates
        return logits


def init_kv_cache(config: TransformerConfig, batch_size: int):
    """Initialize empty KV cache for all layers."""
    head_dim = config.d_model // config.n_heads
    dtype = config.jax_dtype
    cache = []
    for _ in range(config.n_layers):
        cache.append({
            "k": jnp.zeros((batch_size, config.n_heads, config.max_seq_len, head_dim), dtype=dtype),
            "v": jnp.zeros((batch_size, config.n_heads, config.max_seq_len, head_dim), dtype=dtype),
        })
    return cache
