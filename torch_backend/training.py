"""PyTorch training loop for GPT-2 Sudoku model."""

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import SudokuDataset, PAD_TOKEN, SEP_TOKEN, collate_batch
from torch_backend.transformer import GPT2Model, TransformerConfig
from common import TrainConfig, TrainLogger, parse_args


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_schedule(optimizer, cfg: TrainConfig, total_steps: int, warmup_steps: int):
    """Warmup + cosine decay to 0.1 * lr."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine  # decays from 1.0 to 0.1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg: TrainConfig):
    # Seed everything
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = get_device()
    print(f"Using device: {device}", flush=True)

    # Load dataset
    dataset = SudokuDataset(cfg.traces_path)
    n = len(dataset)
    n_val = max(1, int(n * cfg.val_fraction))
    n_train = n - n_val
    all_indices = np.arange(n)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]
    print(f"Dataset: {n} total, {n_train} train, {n_val} val", flush=True)

    max_seq_len = max(len(dataset[i]) for i in range(n))
    print(f"Max sequence length in dataset: {max_seq_len}", flush=True)

    # Compute token budget and steps
    tokens_per_step = cfg.batch_size * (max_seq_len - 1)
    total_steps = cfg.num_tokens // tokens_per_step
    warmup_steps = cfg.warmup_tokens // tokens_per_step
    print(f"Token budget: {cfg.num_tokens:,} tokens -> {total_steps:,} steps ({tokens_per_step} tok/step)", flush=True)

    # Model
    model_cfg = TransformerConfig(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        max_seq_len=max_seq_len,
        dtype=cfg.dtype,
    )
    model = GPT2Model(model_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}", flush=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_schedule(optimizer, cfg, total_steps, warmup_steps)

    # Resume
    start_step = 0
    ckpt_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    if cfg.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(f"Resumed from checkpoint at step {start_step}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Logger
    logger = TrainLogger(log_path=cfg.log_path)
    logger.total_tokens = start_step * tokens_per_step

    print("Starting training...", flush=True)
    pbar = tqdm(
        total=cfg.num_tokens,
        initial=logger.total_tokens,
        unit="tok",
        unit_scale=True,
        desc="Training",
        smoothing=0.1,
    )

    model.train()
    for step in range(start_step, total_steps):
        batch_idx = np.random.choice(train_indices, size=cfg.batch_size, replace=False)
        batch = torch.from_numpy(collate_batch(dataset, batch_idx)).to(device)

        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Mask: only compute loss for tokens after <sep>, excluding pad
        sep_pos = (inputs == SEP_TOKEN).int().argmax(dim=1, keepdim=True)  # (B, 1)
        positions = torch.arange(targets.shape[1], device=device).unsqueeze(0)  # (1, T-1)
        mask = ((positions >= sep_pos) & (targets != PAD_TOKEN)).float()

        logits = model(inputs)
        # Loss in float32
        logits_f32 = logits.float()
        # (B, T-1, V) -> (B*T-1, V) for cross_entropy
        B, T_minus1, V = logits_f32.shape
        per_token_loss = F.cross_entropy(
            logits_f32.reshape(-1, V), targets.reshape(-1), reduction="none"
        ).reshape(B, T_minus1)
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.log_step(step, loss.item(), tokens_per_step)
        pbar.update(tokens_per_step)

        if (step + 1) % cfg.log_every == 0:
            tok_s = logger.tokens_per_sec
            pbar.set_postfix_str(
                f"loss={loss.item():.4f} | tok/s={tok_s / 1000:.1f}K | step={step + 1}"
            )

        if (step + 1) % cfg.val_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(min(10, max(1, n_val // cfg.batch_size))):
                    vi = np.random.choice(val_indices, size=min(cfg.batch_size, n_val), replace=False)
                    vb = torch.from_numpy(collate_batch(dataset, vi)).to(device)
                    v_inputs = vb[:, :-1]
                    v_targets = vb[:, 1:]
                    v_sep = (v_inputs == SEP_TOKEN).int().argmax(dim=1, keepdim=True)
                    v_pos = torch.arange(v_targets.shape[1], device=device).unsqueeze(0)
                    v_mask = ((v_pos >= v_sep) & (v_targets != PAD_TOKEN)).float()
                    v_logits = model(v_inputs).float()
                    vB, vT, vV = v_logits.shape
                    v_ptl = F.cross_entropy(
                        v_logits.reshape(-1, vV), v_targets.reshape(-1), reduction="none"
                    ).reshape(vB, vT)
                    vl = (v_ptl * v_mask).sum() / v_mask.sum().clamp(min=1.0)
                    val_losses.append(vl.item())
            avg_val = np.mean(val_losses)
            logger.log_val(step + 1, float(avg_val))
            logger.save()
            tqdm.write(f"  step {step + 1:>6d} | val_loss {avg_val:.4f}")
            model.train()

        if (step + 1) % cfg.ckpt_every == 0:
            torch.save({
                "step": step + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model_cfg": model_cfg,
            }, ckpt_path)
            logger.log_checkpoint(step + 1)
            tqdm.write(f"  checkpoint saved at step {step + 1}")

    pbar.close()

    # Final checkpoint
    torch.save({
        "step": total_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_cfg": model_cfg,
    }, ckpt_path)

    logger.save()

    summary = logger.summary()
    print(f"\nTraining complete. Summary:")
    print(f"  Steps: {summary['total_steps']:,}")
    print(f"  Tokens: {summary['total_tokens']:,}")
    print(f"  Wall time: {summary['wall_time_s']:.1f}s")
    print(f"  Throughput: {summary['tokens_per_sec'] / 1000:.1f}K tok/s")
    if summary['final_loss'] is not None:
        print(f"  Final train loss: {summary['final_loss']:.4f}")
    if summary['final_val_loss'] is not None:
        print(f"  Final val loss: {summary['final_val_loss']:.4f}")


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
