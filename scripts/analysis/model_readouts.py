"""Model readout and attribution helpers for analysis notebooks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def final_ln_np(x: np.ndarray, params, eps: float = 1e-6) -> np.ndarray:
    scale = np.asarray(params["LayerNorm_0"]["scale"], dtype=np.float32)
    bias = np.asarray(params["LayerNorm_0"]["bias"], dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps) * scale + bias


def final_logits_np(x: np.ndarray, params) -> np.ndarray:
    w = np.asarray(params["lm_head"]["kernel"], dtype=np.float32)
    b = np.asarray(params["lm_head"]["bias"], dtype=np.float32)
    return final_ln_np(x, params) @ w + b


def softmax_fill(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits[..., :729], dtype=np.float64)
    x -= x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def layer7_mlp_parts(params, layer: int = 7) -> dict[str, np.ndarray]:
    return {
        "w_in": np.asarray(params[f"block_{layer}"]["Dense_0"]["kernel"], dtype=np.float32),
        "b_in": np.asarray(params[f"block_{layer}"]["Dense_0"]["bias"], dtype=np.float32),
        "w_out": np.asarray(params[f"block_{layer}"]["Dense_1"]["kernel"], dtype=np.float32),
        "b_out": np.asarray(params[f"block_{layer}"]["Dense_1"]["bias"], dtype=np.float32),
        "ln_scale": np.asarray(params[f"block_{layer}"]["LayerNorm_1"]["scale"], dtype=np.float32),
        "ln_bias": np.asarray(params[f"block_{layer}"]["LayerNorm_1"]["bias"], dtype=np.float32),
        "w_u": np.asarray(params["lm_head"]["kernel"], dtype=np.float32)[:, :729],
    }


def ln_np(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps) * scale + bias


def gelu_np(x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.nn.gelu(jnp.asarray(x)), dtype=np.float32)


def mlp_activations(x_post_attn: np.ndarray, parts: dict[str, np.ndarray]) -> np.ndarray:
    h = ln_np(x_post_attn, parts["ln_scale"], parts["ln_bias"])
    return gelu_np(h @ parts["w_in"] + parts["b_in"])


def neuron_to_token(parts: dict[str, np.ndarray]) -> np.ndarray:
    return parts["w_out"] @ parts["w_u"]


def cell_writer_metadata(parts: dict[str, np.ndarray], score_threshold: float = 1.0, digit_std_threshold: float = 0.2):
    nt = neuron_to_token(parts)
    by_cell = nt.reshape(nt.shape[0], 81, 9)
    cell_mean = by_cell.mean(axis=2)
    cell_std = by_cell.std(axis=2)
    pref_cell = cell_mean.argmax(axis=1)
    pref_score = cell_mean[np.arange(nt.shape[0]), pref_cell]
    pref_digit_std = cell_std[np.arange(nt.shape[0]), pref_cell]
    mask = (pref_score > score_threshold) & (pref_digit_std < digit_std_threshold)
    return {
        "neuron_to_token": nt,
        "pref_cell": pref_cell,
        "pref_score": pref_score,
        "pref_digit_std": pref_digit_std,
        "cell_writer_mask": mask,
    }


def make_patched_fn(model, params, layer_idx: int):
    @jax.jit
    def fn(seq_arr, pos, delta):
        return model.apply({"params": params}, seq_arr, patch={"layer": layer_idx, "pos": pos, "delta": delta})

    return fn

