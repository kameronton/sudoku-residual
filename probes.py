"""Linear probing for Sudoku transformer residual stream analysis."""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from tqdm import tqdm

from data import SEP_TOKEN, PAD_TOKEN, MAX_SEQ_LEN, encode_fill, decode_fill
from model import GPT2Model, TransformerConfig
from training import make_schedule, TrainConfig, encode_clues


# ---------------------------------------------------------------------------
# Feature extraction for Sudoku boards
# ---------------------------------------------------------------------------

def get_candidates(puzzle: str, pos: int) -> set[int]:
    """Get all legal candidate digits for a position given the current puzzle state."""
    if puzzle[pos] in "123456789":
        return set()  # Already filled
    
    r, c = divmod(pos, 9)
    used = set()
    
    # Row constraint
    for j in range(9):
        ch = puzzle[r * 9 + j]
        if ch in "123456789":
            used.add(int(ch))
    
    # Column constraint
    for i in range(9):
        ch = puzzle[i * 9 + c]
        if ch in "123456789":
            used.add(int(ch))
    
    # Box constraint
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            ch = puzzle[i * 9 + j]
            if ch in "123456789":
                used.add(int(ch))
    
    return set(range(1, 10)) - used


@dataclass
class SudokuFeatures:
    """Features extracted from a Sudoku puzzle state."""
    
    # Binary mask: 1 if cell is filled (clue), 0 if empty (81,)
    filled_mask: np.ndarray
    
    # Digit at each position: 0 for empty, 1-9 for filled (81,)
    digits: np.ndarray
    
    # One-hot digit encoding: (81, 9) - position [i, d-1] = 1 if cell i has digit d
    digits_onehot: np.ndarray
    
    # Candidate sets as binary: (81, 9) - position [i, d-1] = 1 if digit d is a candidate for cell i
    candidates: np.ndarray
    
    # Number of candidates per cell (81,)
    n_candidates: np.ndarray
    
    # Row, column, box indices for each position (81, 3)
    position_info: np.ndarray


def extract_features(puzzle: str) -> SudokuFeatures:
    """Extract all features from a puzzle string."""
    filled_mask = np.zeros(81, dtype=np.float32)
    digits = np.zeros(81, dtype=np.float32)
    digits_onehot = np.zeros((81, 9), dtype=np.float32)
    candidates = np.zeros((81, 9), dtype=np.float32)
    n_candidates = np.zeros(81, dtype=np.float32)
    position_info = np.zeros((81, 3), dtype=np.float32)
    
    for pos in range(81):
        r, c = divmod(pos, 9)
        box = (r // 3) * 3 + (c // 3)
        position_info[pos] = [r, c, box]
        
        ch = puzzle[pos]
        if ch in "123456789":
            filled_mask[pos] = 1.0
            d = int(ch)
            digits[pos] = d
            digits_onehot[pos, d - 1] = 1.0
        else:
            cands = get_candidates(puzzle, pos)
            for d in cands:
                candidates[pos, d - 1] = 1.0
            n_candidates[pos] = len(cands)
    
    return SudokuFeatures(
        filled_mask=filled_mask,
        digits=digits,
        digits_onehot=digits_onehot,
        candidates=candidates,
        n_candidates=n_candidates,
        position_info=position_info,
    )


def flatten_features(features: SudokuFeatures) -> np.ndarray:
    """Flatten all features into a single vector for probing."""
    return np.concatenate([
        features.filled_mask,           # 81
        features.digits / 9.0,          # 81 (normalized)
        features.digits_onehot.flatten(),  # 81 * 9 = 729
        features.candidates.flatten(),    # 81 * 9 = 729
        features.n_candidates / 9.0,      # 81 (normalized)
    ])
    # Total: 81 + 81 + 729 + 729 + 81 = 1701 features


# ---------------------------------------------------------------------------
# Model loading and activation collection
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_dir: str, model_cfg: TransformerConfig):
    """Load model from checkpoint directory."""
    model = GPT2Model(model_cfg)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]

    train_cfg = TrainConfig()
    schedule = make_schedule(train_cfg, total_steps=1, warmup_steps=0)
    tx = optax.adamw(learning_rate=schedule, weight_decay=train_cfg.weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    ckpt_mgr = ocp.CheckpointManager(os.path.abspath(ckpt_dir))
    step = ckpt_mgr.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    state = ckpt_mgr.restore(step, args=ocp.args.StandardRestore(state))
    print(f"Loaded checkpoint at step {int(state.step)}", flush=True)
    return state.params, model


def make_forward_with_intermediates(model: GPT2Model):
    """Create a JIT-compiled forward function that returns intermediate activations."""
    @jax.jit
    def forward(params, tokens):
        logits, intermediates = model.apply(
            {"params": params}, tokens, return_intermediates=True
        )
        return logits, intermediates
    return forward


def collect_activations(
    forward_fn: Callable,
    params,
    puzzles: list[str],
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[SudokuFeatures]]:
    """
    Collect residual stream activations for a list of puzzles.
    
    Returns:
        activations: (n_puzzles, n_layers, seq_len, d_model) array of activations
        sep_positions: (n_puzzles,) array of separator token positions
        features: list of SudokuFeatures for each puzzle
    """
    all_activations = []
    all_sep_positions = []
    all_features = []
    
    for i in tqdm(range(0, len(puzzles), batch_size), desc="Collecting activations"):
        batch_puzzles = puzzles[i:i + batch_size]
        
        # Tokenize puzzles
        batch_tokens = []
        batch_sep_pos = []
        for puzzle in batch_puzzles:
            tokens = encode_clues(puzzle)
            batch_sep_pos.append(len(tokens) - 1)  # Position of SEP token
            # Pad to MAX_SEQ_LEN
            tokens = tokens + [PAD_TOKEN] * (MAX_SEQ_LEN - len(tokens))
            batch_tokens.append(tokens)
        
        # Run forward pass
        tokens_arr = jnp.array(batch_tokens, dtype=jnp.int32)
        _, intermediates = forward_fn(params, tokens_arr)
        
        # intermediates: (n_layers, batch, seq_len, d_model)
        # Transpose to (batch, n_layers, seq_len, d_model)
        intermediates = jnp.transpose(intermediates, (1, 0, 2, 3))
        all_activations.append(np.array(intermediates))
        all_sep_positions.extend(batch_sep_pos)
        
        # Extract features for each puzzle
        for puzzle in batch_puzzles:
            all_features.append(extract_features(puzzle))
    
    activations = np.concatenate(all_activations, axis=0)
    sep_positions = np.array(all_sep_positions)
    
    return activations, sep_positions, all_features


# ---------------------------------------------------------------------------
# Linear probe training
# ---------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    """Configuration for linear probe training."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 100
    batch_size: int = 128
    val_fraction: float = 0.1


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    config: ProbeConfig,
    regression: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Train a linear probe on activations X to predict targets y.
    
    Args:
        X: (n_samples, d_model) activations
        y: (n_samples,) or (n_samples, n_targets) targets
        config: ProbeConfig
        regression: If True, use MSE loss. If False, use cross-entropy.
    
    Returns:
        weights: Trained probe weights
        metrics: Dictionary of training metrics
    """
    n_samples = X.shape[0]
    input_dim = X.shape[1]
    
    # Handle multi-dimensional targets
    if y.ndim == 1:
        output_dim = 1
        y = y.reshape(-1, 1)
    else:
        output_dim = y.shape[1]
    
    # Train/val split
    n_val = max(1, int(n_samples * config.val_fraction))
    n_train = n_samples - n_val
    
    perm = np.random.permutation(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Initialize probe parameters (weight matrix + bias)
    rng = jax.random.PRNGKey(42)
    W = jax.random.normal(rng, (input_dim, output_dim)) * 0.01
    b = jnp.zeros(output_dim)
    params = {"W": W, "b": b}
    
    # Optimizer
    tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    opt_state = tx.init(params)
    
    def predict(params, x):
        return x @ params["W"] + params["b"]
    
    def mse_loss(params, x, y):
        pred = predict(params, x)
        return jnp.mean((pred - y) ** 2)
    
    def cross_entropy_loss(params, x, y):
        logits = predict(params, x)
        # y is one-hot or class indices
        if y.shape[-1] == logits.shape[-1]:
            # One-hot targets
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.mean(jnp.sum(y * log_probs, axis=-1))
        else:
            # Class indices
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.mean(log_probs[jnp.arange(len(y)), y.squeeze().astype(int)])
    
    loss_fn = mse_loss if regression else cross_entropy_loss
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Training loop
    train_losses = []
    val_losses = []
    
    X_train_jax = jnp.array(X_train)
    y_train_jax = jnp.array(y_train)
    X_val_jax = jnp.array(X_val)
    y_val_jax = jnp.array(y_val)
    
    for epoch in range(config.n_epochs):
        # Shuffle training data
        perm = np.random.permutation(n_train)
        epoch_losses = []
        
        for i in range(0, n_train, config.batch_size):
            batch_idx = perm[i:i + config.batch_size]
            x_batch = X_train_jax[batch_idx]
            y_batch = y_train_jax[batch_idx]
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(float(loss))
        
        train_loss = np.mean(epoch_losses)
        val_loss = float(loss_fn(params, X_val_jax, y_val_jax))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config.n_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    # Extract final weights
    weights = np.array(params["W"])
    bias = np.array(params["b"])
    
    # Compute final metrics
    if regression:
        train_pred = np.array(predict(params, X_train_jax))
        val_pred = np.array(predict(params, X_val_jax))
        
        # R² score
        def r2_score(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
            return 1 - ss_res / (ss_tot + 1e-8)
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
    else:
        train_pred = np.argmax(np.array(predict(params, X_train_jax)), axis=-1)
        val_pred = np.argmax(np.array(predict(params, X_val_jax)), axis=-1)
        
        if y_train.shape[-1] > 1:
            y_train_cls = np.argmax(y_train, axis=-1)
            y_val_cls = np.argmax(y_val, axis=-1)
        else:
            y_train_cls = y_train.squeeze()
            y_val_cls = y_val.squeeze()
        
        train_r2 = np.mean(train_pred == y_train_cls)
        val_r2 = np.mean(val_pred == y_val_cls)
    
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_r2": float(train_r2),
        "val_r2": float(val_r2),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
    }
    
    return {"W": weights, "b": bias}, metrics


# ---------------------------------------------------------------------------
# Probing experiments
# ---------------------------------------------------------------------------

def get_activations_for_position(
    activations: np.ndarray,
    layer: int,
    position: str,
    sep_positions: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract activations at specified position(s).
    
    Args:
        activations: (n_samples, n_layers, seq_len, d_model)
        layer: Which layer
        position: "sep", "mean_clues", "last", or int
        sep_positions: Required for "sep" and "mean_clues"
    
    Returns:
        X: (n_samples, d_model) or (n_samples, d_model * multiplier) for concat modes
    """
    n_samples = activations.shape[0]
    
    if position == "sep":
        assert sep_positions is not None
        X = np.array([activations[i, layer, sep_positions[i], :] for i in range(n_samples)])
    elif position == "mean_clues":
        # Average over all clue token positions (0 to sep_pos-1)
        assert sep_positions is not None
        X = np.array([
            activations[i, layer, :sep_positions[i], :].mean(axis=0)
            for i in range(n_samples)
        ])
    elif position == "concat_stats":
        # Concatenate mean, max, and SEP token activations
        assert sep_positions is not None
        X_list = []
        for i in range(n_samples):
            clue_acts = activations[i, layer, :sep_positions[i], :]
            sep_act = activations[i, layer, sep_positions[i], :]
            mean_act = clue_acts.mean(axis=0)
            max_act = clue_acts.max(axis=0)
            X_list.append(np.concatenate([mean_act, max_act, sep_act]))
        X = np.array(X_list)
    elif position == "last":
        X = activations[:, layer, -1, :]
    else:
        X = activations[:, layer, int(position), :]
    
    return X


def probe_filled_mask(
    activations: np.ndarray,
    features: list[SudokuFeatures],
    layer: int,
    position: str = "sep",
    sep_positions: np.ndarray | None = None,
    config: ProbeConfig | None = None,
) -> tuple[dict, dict]:
    """
    Probe for the filled/empty mask of the initial puzzle.
    
    Args:
        activations: (n_samples, n_layers, seq_len, d_model)
        features: List of SudokuFeatures
        layer: Which layer to probe (0-indexed, -1 for last)
        position: "sep", "mean_clues", "concat_stats", "last", or int
        sep_positions: Required if position uses clue info
    """
    if config is None:
        config = ProbeConfig()
    
    X = get_activations_for_position(activations, layer, position, sep_positions)
    
    # Target: filled mask (81 values per puzzle)
    y = np.array([f.filled_mask for f in features])
    
    print(f"\nProbing filled mask from layer {layer}, position={position}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    weights, metrics = train_linear_probe(X, y, config, regression=True)
    
    print(f"Train R²: {metrics['train_r2']:.4f}, Val R²: {metrics['val_r2']:.4f}")
    
    return weights, metrics


def probe_digit_prediction(
    activations: np.ndarray,
    features: list[SudokuFeatures],
    layer: int,
    cell_idx: int,
    sep_positions: np.ndarray,
    position: str = "sep",
    config: ProbeConfig | None = None,
) -> tuple[dict, dict]:
    """
    Probe for the digit at a specific cell position.
    Only uses puzzles where that cell is filled.
    """
    if config is None:
        config = ProbeConfig()
    
    n_samples = activations.shape[0]
    
    # Filter to puzzles where cell_idx is filled
    valid_indices = [i for i in range(n_samples) if features[i].filled_mask[cell_idx] > 0.5]
    
    if len(valid_indices) < 100:
        print(f"Warning: Only {len(valid_indices)} samples have cell {cell_idx} filled")
        return None, {"error": "insufficient_samples"}
    
    # Get activations for valid samples
    valid_activations = activations[valid_indices]
    valid_sep_positions = sep_positions[valid_indices]
    
    X = get_activations_for_position(valid_activations, layer, position, valid_sep_positions)
    y = np.array([features[i].digits_onehot[cell_idx] for i in valid_indices])
    
    print(f"\nProbing digit at cell {cell_idx} from layer {layer}, position={position}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    weights, metrics = train_linear_probe(X, y, config, regression=False)
    
    print(f"Train accuracy: {metrics['train_r2']:.4f}, Val accuracy: {metrics['val_r2']:.4f}")
    
    return weights, metrics


def probe_all_digits(
    activations: np.ndarray,
    features: list[SudokuFeatures],
    layer: int,
    sep_positions: np.ndarray,
    position: str = "sep",
    config: ProbeConfig | None = None,
) -> tuple[dict, dict]:
    """
    Probe for all 81 digit values simultaneously.
    Target: (n_samples, 81) with values 0-9 (0 for empty, 1-9 for filled digits)
    """
    if config is None:
        config = ProbeConfig()
    
    X = get_activations_for_position(activations, layer, position, sep_positions)
    
    # Target: all digits (81 per puzzle), normalized
    y = np.array([f.digits for f in features]) / 9.0
    
    print(f"\nProbing all digits from layer {layer}, position={position}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    weights, metrics = train_linear_probe(X, y, config, regression=True)
    
    print(f"Train R²: {metrics['train_r2']:.4f}, Val R²: {metrics['val_r2']:.4f}")
    
    return weights, metrics


def probe_candidates(
    activations: np.ndarray,
    features: list[SudokuFeatures],
    layer: int,
    sep_positions: np.ndarray,
    position: str = "sep",
    config: ProbeConfig | None = None,
) -> tuple[dict, dict]:
    """
    Probe for candidate sets (which digits are possible for each empty cell).
    Target: (n_samples, 81*9) binary array
    """
    if config is None:
        config = ProbeConfig()
    
    X = get_activations_for_position(activations, layer, position, sep_positions)
    
    # Target: candidate sets flattened
    y = np.array([f.candidates.flatten() for f in features])
    
    print(f"\nProbing candidate sets from layer {layer}, position={position}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    weights, metrics = train_linear_probe(X, y, config, regression=True)
    
    print(f"Train R²: {metrics['train_r2']:.4f}, Val R²: {metrics['val_r2']:.4f}")
    
    return weights, metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Linear probing for Sudoku transformer")
    parser.add_argument("--ckpt_dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--data_path", default="sudoku-3m.csv", help="Path to puzzle data")
    parser.add_argument("--n_puzzles", type=int, default=1000, help="Number of puzzles to use")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in CSV")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for activation collection")
    parser.add_argument("--probe_layer", type=int, default=-1, help="Layer to probe (-1 for last)")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in model")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=512, help="FFN dimension")
    parser.add_argument("--probe_epochs", type=int, default=100, help="Probe training epochs")
    parser.add_argument("--save_activations", type=str, default=None, help="Save activations to file")
    parser.add_argument("--load_activations", type=str, default=None, help="Load activations from file")
    args = parser.parse_args()
    
    # Model config
    model_cfg = TransformerConfig(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
    )
    
    # Probe config
    probe_cfg = ProbeConfig(n_epochs=args.probe_epochs)
    
    if args.load_activations:
        print(f"Loading activations from {args.load_activations}")
        data = np.load(args.load_activations, allow_pickle=True)
        activations = data["activations"]
        sep_positions = data["sep_positions"]
        puzzles = list(data["puzzles"])
        features = [extract_features(p) for p in puzzles]
    else:
        # 1. Load model
        print("Loading model...")
        params, model = load_checkpoint(args.ckpt_dir, model_cfg)
        forward_fn = make_forward_with_intermediates(model)
        
        # 2. Load puzzles
        print(f"Loading puzzles from {args.data_path}...")
        puzzles = []
        with open(args.data_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i < args.offset:
                    continue
                puzzles.append(row["puzzle"])
                if len(puzzles) >= args.n_puzzles:
                    break
        
        print(f"Loaded {len(puzzles)} puzzles")
        
        # 3. Collect activations
        print("Collecting activations...")
        activations, sep_positions, features = collect_activations(
            forward_fn, params, puzzles, batch_size=args.batch_size
        )
        
        if args.save_activations:
            print(f"Saving activations to {args.save_activations}")
            np.savez_compressed(
                args.save_activations,
                activations=activations,
                sep_positions=sep_positions,
                puzzles=np.array(puzzles),
            )
    
    print(f"\nActivations shape: {activations.shape}")
    print(f"  (n_puzzles={activations.shape[0]}, n_layers={activations.shape[1]}, "
          f"seq_len={activations.shape[2]}, d_model={activations.shape[3]})")
    
    # 4. Run probing experiments
    layer = args.probe_layer
    if layer < 0:
        layer = activations.shape[1] + layer
    
    print(f"\n{'='*60}")
    print(f"Comparing aggregation strategies on layer {layer}")
    print(f"{'='*60}")
    
    # Compare different ways to aggregate activations
    for position in ["sep", "mean_clues", "concat_stats"]:
        print(f"\n--- Position: {position} ---")
        
        # Probe 1: Filled mask
        _, filled_metrics = probe_filled_mask(
            activations, features, layer, position, sep_positions, probe_cfg
        )
        
        # Probe 2: All digits
        _, digits_metrics = probe_all_digits(
            activations, features, layer, sep_positions, position, probe_cfg
        )
        
        # Probe 3: Candidate sets
        _, candidates_metrics = probe_candidates(
            activations, features, layer, sep_positions, position, probe_cfg
        )
    
    # Run probes across all layers with best aggregation strategy
    print(f"\n{'='*60}")
    print(f"Probing filled mask across all layers (mean_clues aggregation)")
    print(f"{'='*60}")
    
    layer_results = []
    for l in range(activations.shape[1]):
        _, metrics = probe_filled_mask(
            activations, features, l, "mean_clues", sep_positions,
            ProbeConfig(n_epochs=50, batch_size=128)  # Faster for comparison
        )
        layer_results.append({
            "layer": l,
            "train_r2": metrics["train_r2"],
            "val_r2": metrics["val_r2"],
        })
    
    print(f"\nLayer-wise R² scores for filled mask:")
    for r in layer_results:
        print(f"  Layer {r['layer']}: train={r['train_r2']:.4f}, val={r['val_r2']:.4f}")


# ---------------------------------------------------------------------------
# Simple single-cell probe experiment
# ---------------------------------------------------------------------------

def is_cell_filled_in_sequence(seq: np.ndarray, cell_row: int, cell_col: int, sep_pos: int) -> bool:
    """Check if a specific cell is filled (clue) in a tokenized sequence."""
    # Tokens for cell (r, c) are: r*81 + c*9 + (d-1) for d in 1..9
    # So tokens r*81 + c*9 + 0 through r*81 + c*9 + 8
    base_token = cell_row * 81 + cell_col * 9
    cell_tokens = set(range(base_token, base_token + 9))
    
    # Check clue portion (before SEP)
    for t in seq[:sep_pos]:
        if t in cell_tokens:
            return True
    return False


def probe_single_cell_binary(
    activations: np.ndarray,
    labels: np.ndarray,
    layer: int,
    position_indices: np.ndarray,
    position_type: str = "sep",
    config: ProbeConfig | None = None,
) -> tuple[dict, dict]:
    """
    Train a logistic probe to predict a binary label from activations.
    
    Args:
        activations: (n_samples, n_layers, seq_len, d_model)
        labels: (n_samples,) binary labels
        layer: Which layer to probe
        position_indices: (n_samples,) indices for position extraction
        position_type: Description for logging
    """
    if config is None:
        config = ProbeConfig()
    
    n_samples = activations.shape[0]
    
    # Extract activations at specified positions
    X = np.array([activations[i, layer, position_indices[i], :] for i in range(n_samples)])
    y = labels.astype(np.float32).reshape(-1, 1)
    
    # Train/val split
    n_val = max(1, int(n_samples * config.val_fraction))
    n_train = n_samples - n_val
    
    perm = np.random.permutation(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Initialize logistic probe
    input_dim = X.shape[1]
    rng = jax.random.PRNGKey(42)
    W = jax.random.normal(rng, (input_dim, 1)) * 0.01
    b = jnp.zeros(1)
    params = {"W": W, "b": b}
    
    tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    opt_state = tx.init(params)
    
    def predict_logits(params, x):
        return x @ params["W"] + params["b"]
    
    def bce_loss(params, x, y):
        logits = predict_logits(params, x)
        # Binary cross-entropy with logits (numerically stable)
        # BCE = max(logits, 0) - logits*y + log(1 + exp(-|logits|))
        return jnp.mean(jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(bce_loss)(params, x, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    X_train_jax = jnp.array(X_train)
    y_train_jax = jnp.array(y_train)
    X_val_jax = jnp.array(X_val)
    y_val_jax = jnp.array(y_val)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config.n_epochs):
        perm = np.random.permutation(n_train)
        epoch_losses = []
        
        for i in range(0, n_train, config.batch_size):
            batch_idx = perm[i:i + config.batch_size]
            x_batch = X_train_jax[batch_idx]
            y_batch = y_train_jax[batch_idx]
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(float(loss))
        
        train_loss = np.mean(epoch_losses)
        val_loss = float(bce_loss(params, X_val_jax, y_val_jax))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{config.n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Compute accuracy
    train_preds = (np.array(jax.nn.sigmoid(predict_logits(params, X_train_jax))) > 0.5).astype(float)
    val_preds = (np.array(jax.nn.sigmoid(predict_logits(params, X_val_jax))) > 0.5).astype(float)
    
    train_acc = np.mean(train_preds == y_train)
    val_acc = np.mean(val_preds == y_val)
    
    # Class balance
    train_pos_rate = np.mean(y_train)
    val_pos_rate = np.mean(y_val)
    
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "train_pos_rate": float(train_pos_rate),
        "val_pos_rate": float(val_pos_rate),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
    }
    
    weights = {"W": np.array(params["W"]), "b": np.array(params["b"])}
    
    return weights, metrics


def run_single_cell_experiment():
    """Run focused experiment: can we detect if cell (0,0) is filled?"""
    parser = argparse.ArgumentParser(description="Probe for single cell filled/empty")
    parser.add_argument("--traces_path", default="traces_constraint_fixed.npz")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--probe_epochs", type=int, default=100)
    parser.add_argument("--cell_row", type=int, default=0)
    parser.add_argument("--cell_col", type=int, default=0)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=512)
    args = parser.parse_args()
    
    print(f"Probing for cell ({args.cell_row}, {args.cell_col}) filled/empty")
    
    # 1. Load sequences from traces file
    print(f"\nLoading sequences from {args.traces_path}...")
    data = np.load(args.traces_path)
    sequences = data["sequences"][:args.n_samples]
    n_samples = len(sequences)
    print(f"Loaded {n_samples} sequences")
    
    # 2. Extract labels and SEP positions
    labels = []
    sep_positions = []
    
    for seq in sequences:
        sep_pos = np.where(seq == SEP_TOKEN)[0][0]
        sep_positions.append(sep_pos)
        is_filled = is_cell_filled_in_sequence(seq, args.cell_row, args.cell_col, sep_pos)
        labels.append(is_filled)
    
    labels = np.array(labels, dtype=np.float32)
    sep_positions = np.array(sep_positions)
    
    pos_rate = np.mean(labels)
    print(f"Cell ({args.cell_row}, {args.cell_col}) filled rate: {pos_rate:.2%}")
    
    # 3. Load model
    print("\nLoading model...")
    model_cfg = TransformerConfig(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
    )
    params, model = load_checkpoint(args.ckpt_dir, model_cfg)
    forward_fn = make_forward_with_intermediates(model)
    
    # 4. Collect activations
    print("\nCollecting activations...")
    all_activations = []
    
    for i in tqdm(range(0, n_samples, args.batch_size), desc="Forward pass"):
        batch_seqs = sequences[i:i + args.batch_size]
        tokens_arr = jnp.array(batch_seqs, dtype=jnp.int32)
        _, intermediates = forward_fn(params, tokens_arr)
        # intermediates: (n_layers, batch, seq_len, d_model)
        intermediates = jnp.transpose(intermediates, (1, 0, 2, 3))
        all_activations.append(np.array(intermediates))
    
    activations = np.concatenate(all_activations, axis=0)
    print(f"Activations shape: {activations.shape}")
    
    # 5. Run probes
    probe_cfg = ProbeConfig(n_epochs=args.probe_epochs)
    n_layers = activations.shape[1]
    
    print(f"\n{'='*60}")
    print(f"Probing: Is cell ({args.cell_row}, {args.cell_col}) filled?")
    print(f"{'='*60}")
    
    # Probe from SEP token at each layer
    print("\n--- Probing from SEP token ---")
    layer_results_sep = []
    for layer in range(n_layers):
        print(f"\nLayer {layer}:")
        _, metrics = probe_single_cell_binary(
            activations, labels, layer, sep_positions, "sep", probe_cfg
        )
        print(f"  Train acc: {metrics['train_acc']:.4f}, Val acc: {metrics['val_acc']:.4f}")
        print(f"  (Baseline: {max(metrics['val_pos_rate'], 1-metrics['val_pos_rate']):.4f})")
        layer_results_sep.append({"layer": layer, **metrics})
    
    # Probe from position 0 (first token) at each layer
    print("\n--- Probing from position 0 (first clue token) ---")
    first_positions = np.zeros(n_samples, dtype=np.int32)
    layer_results_pos0 = []
    for layer in range(n_layers):
        print(f"\nLayer {layer}:")
        _, metrics = probe_single_cell_binary(
            activations, labels, layer, first_positions, "pos0", probe_cfg
        )
        print(f"  Train acc: {metrics['train_acc']:.4f}, Val acc: {metrics['val_acc']:.4f}")
        layer_results_pos0.append({"layer": layer, **metrics})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Validation Accuracy by Layer")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'SEP Token':<15} {'Position 0':<15}")
    print("-" * 40)
    for i in range(n_layers):
        sep_acc = layer_results_sep[i]['val_acc']
        pos0_acc = layer_results_pos0[i]['val_acc']
        print(f"{i:<8} {sep_acc:<15.4f} {pos0_acc:<15.4f}")
    
    baseline = max(pos_rate, 1 - pos_rate)
    print(f"\nBaseline (majority class): {baseline:.4f}")


def run_multi_cell_experiment():
    """Run probing for multiple cells at layer 5 SEP token."""
    parser = argparse.ArgumentParser(description="Probe for multiple cells")
    parser.add_argument("--traces_path", default="traces_constraint_fixed.npz")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=512)
    args = parser.parse_args()
    
    # Cells to test: corners, center, edges
    cells_to_test = [
        (0, 0), (0, 4), (0, 8),  # Top row
        (4, 0), (4, 4), (4, 8),  # Middle row  
        (8, 0), (8, 4), (8, 8),  # Bottom row
    ]
    
    # 1. Load sequences
    print(f"Loading sequences from {args.traces_path}...")
    data = np.load(args.traces_path)
    sequences = data["sequences"][:args.n_samples]
    n_samples = len(sequences)
    print(f"Loaded {n_samples} sequences")
    
    # 2. Extract SEP positions
    sep_positions = np.array([np.where(seq == SEP_TOKEN)[0][0] for seq in sequences])
    
    # 3. Load model and collect activations
    print("\nLoading model...")
    model_cfg = TransformerConfig(
        n_layers=args.n_layers, n_heads=args.n_heads,
        d_model=args.d_model, d_ff=args.d_ff,
    )
    params, model = load_checkpoint(args.ckpt_dir, model_cfg)
    forward_fn = make_forward_with_intermediates(model)
    
    print("Collecting activations...")
    all_activations = []
    for i in tqdm(range(0, n_samples, args.batch_size), desc="Forward pass"):
        batch_seqs = sequences[i:i + args.batch_size]
        tokens_arr = jnp.array(batch_seqs, dtype=jnp.int32)
        _, intermediates = forward_fn(params, tokens_arr)
        intermediates = jnp.transpose(intermediates, (1, 0, 2, 3))
        all_activations.append(np.array(intermediates))
    activations = np.concatenate(all_activations, axis=0)
    
    # 4. Probe each cell
    probe_cfg = ProbeConfig(n_epochs=args.probe_epochs)
    results = []
    
    print(f"\n{'='*70}")
    print("Probing all cells from SEP token at layer 5")
    print(f"{'='*70}")
    
    for cell_row, cell_col in cells_to_test:
        # Extract labels for this cell
        labels = np.array([
            is_cell_filled_in_sequence(seq, cell_row, cell_col, sep_positions[i])
            for i, seq in enumerate(sequences)
        ], dtype=np.float32)
        
        pos_rate = np.mean(labels)
        baseline = max(pos_rate, 1 - pos_rate)
        
        print(f"\nCell ({cell_row}, {cell_col}): filled_rate={pos_rate:.2%}, baseline={baseline:.2%}")
        
        _, metrics = probe_single_cell_binary(
            activations, labels, layer=5, position_indices=sep_positions,
            position_type="sep", config=probe_cfg
        )
        
        results.append({
            "cell": (cell_row, cell_col),
            "filled_rate": pos_rate,
            "baseline": baseline,
            "val_acc": metrics["val_acc"],
            "improvement": metrics["val_acc"] - baseline,
        })
        print(f"  Val accuracy: {metrics['val_acc']:.4f} (improvement: +{metrics['val_acc'] - baseline:.4f})")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Layer 5 SEP Token Probing Results")
    print(f"{'='*70}")
    print(f"{'Cell':<12} {'Fill Rate':<12} {'Baseline':<12} {'Val Acc':<12} {'Improvement':<12}")
    print("-" * 60)
    for r in results:
        cell_str = f"({r['cell'][0]}, {r['cell'][1]})"
        print(f"{cell_str:<12} {r['filled_rate']:<12.2%} {r['baseline']:<12.2%} {r['val_acc']:<12.4f} +{r['improvement']:<11.4f}")
    
    avg_improvement = np.mean([r["improvement"] for r in results])
    avg_acc = np.mean([r["val_acc"] for r in results])
    print("-" * 60)
    print(f"{'Average':<12} {'':<12} {'':<12} {avg_acc:<12.4f} +{avg_improvement:<11.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--single-cell":
        sys.argv.pop(1)
        run_single_cell_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == "--multi-cell":
        sys.argv.pop(1)
        run_multi_cell_experiment()
    else:
        main()


