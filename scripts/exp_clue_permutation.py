"""permutation of clues invariance test

On the last checkpoint, where the model is supposed to have a structural 
representation of Sudoku, we permute the order of the clue tokens, and we
measure, layer by layer, wheather the residual stream at the SEP pos remains
approximately the same. 

"""
import argparse
import json
import random

import numpy as np

from sudoku.activations import (
    load_checkpoint,
    make_intermediates_fn,
    encode_clues,
    collect_activations,
)


def _as_str(p) -> str:
    return p.decode() if isinstance(p, bytes) else str(p)


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosinus sur le dernier axe ; a, b de forme (..., d)."""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return (a * b).sum(-1) / (na * nb + 1e-9)


def _plot(res: dict, path: str) -> None:
    """Figure : cos(perm), cos(autre) et score d'invariance par couche."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non installe -- figure ignoree (uv pip install matplotlib).")
        return
    layers = [d["layer"] for d in res["layers"]]
    cp = [d["cos_perm"] for d in res["layers"]]
    co = [d["cos_other"] for d in res["layers"]]
    inv = [d["invariance"] for d in res["layers"]]
    plt.figure(figsize=(7, 4.5))
    plt.plot(layers, cp, "o-", label="cos(perm) -- meme grille, ordre permute")
    plt.plot(layers, co, "s--", label="cos(autre) -- autre grille, meme #indices")
    plt.plot(layers, inv, "^:", color="green", label="score d'invariance")
    plt.xlabel("couche")
    plt.ylabel("similarite cosinus")
    plt.ylim(-0.05, 1.05)
    plt.title("Invariance a l'ordre des indices (residual stream, position SEP)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Figure : {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/3M-backtracking-packing/checkpoint")
    ap.add_argument("--traces", default="bt_traces_3m.npz")
    ap.add_argument("--n", type=int, default=256, help="nombre de grilles")
    ap.add_argument("--k", type=int, default=32, help="permutations par grille")
    ap.add_argument("--chunk", type=int, default=32, help="grilles par chunk (mémoire)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="clue_perm_invariance.json")
    ap.add_argument("--plot", default="clue_perm_invariance.png")
    args = ap.parse_args()

    random.seed(0)

    params, model = load_checkpoint(args.ckpt)
    fn = make_intermediates_fn(model)

    tp = np.load(args.traces, allow_pickle=True)
    pkey = "puzzles_test" if "puzzles_test" in tp else "puzzles_val"
    puzzles = [_as_str(p) for p in tp[pkey][: args.n]]
    print(f"{len(puzzles)} grilles, {args.k} permutations chacune", flush=True)

    # On ne garde que le vecteur à la position SEP.
    sep_canon: list[tuple[int, np.ndarray, int]] = []   # (idx grille, (n_layers, d), nb_indices)
    sep_perm: list[np.ndarray] = []                     # (n_layers, d) par permutation
    perm_owner: list[int] = []                          # idx grille de chaque permutation

    for c0 in range(0, len(puzzles), args.chunk):
        chunk = puzzles[c0 : c0 + args.chunk]
        seqs, tag = [], []   # tag = ("canon"|"perm", idx local)
        for li, p in enumerate(chunk):
            seqs.append(encode_clues(p, randomize_order=False, use_sep=True))
            tag.append(("canon", li))
            for _ in range(args.k):
                seqs.append(encode_clues(p, randomize_order=True, use_sep=True))
                tag.append(("perm", li))
        acts = collect_activations(fn, params, seqs, args.batch)["post_mlp"]  
        for m, (kind, li) in enumerate(tag):
            sep = len(seqs[m]) - 1            # position du token SEP
            v = acts[m, :, sep, :]            # (n_layers, d)
            if kind == "canon":
                sep_canon.append((c0 + li, v, sep))   # sep = nb d'indices (= position SEP)
            else:
                sep_perm.append(v)
                perm_owner.append(c0 + li)
        del acts

    n_layers = sep_canon[0][1].shape[0]
    canon_by_idx = {gi: v for gi, v, _ in sep_canon}

    # within-puzzle : cos(canonique, permutation) par couche
    within = np.stack(
        [cos(canon_by_idx[gi], v) for gi, v in zip(perm_owner, sep_perm)], 0
    )  # (n_perm, n_layers)

    # across-puzzle : cos entre grilles DIFFÉRENTES mais de MÊME nombre d'indices
    # (sinon la position du SEP / le compte d'indices biaiserait la baseline).
    canon_arr = np.stack([v for _, v, _ in sep_canon], 0)   # (n, n_layers, d)
    counts = np.array([cnt for _, _, cnt in sep_canon])
    groups: dict[int, list[int]] = {}
    for i, cnt in enumerate(counts):
        groups.setdefault(int(cnt), []).append(i)
    eligible = [np.array(g) for g in groups.values() if len(g) >= 2]
    if not eligible:
        raise RuntimeError("Aucun groupe de >=2 grilles de meme #indices (augmente --n).")
    print(f"baseline appariee : {len(counts)} grilles, {len(groups)} valeurs de #indices, "
          f"{len(eligible)} groupes utilisables (>=2)", flush=True)
    rng = np.random.default_rng(0)
    weights = np.array([len(g) * (len(g) - 1) / 2 for g in eligible], dtype=float)
    probs = weights / weights.sum()   # tirage uniforme sur les paires appariées
    pairs = np.empty((4000, 2), dtype=int)
    for t in range(4000):
        g = eligible[rng.choice(len(eligible), p=probs)]
        i, j = rng.choice(len(g), 2, replace=False)
        pairs[t] = (g[i], g[j])
    across = cos(canon_arr[pairs[:, 0]], canon_arr[pairs[:, 1]])   # (4000, n_layers)

    res = {"n_layers": int(n_layers), "n_puzzles": len(puzzles), "k": args.k, "layers": []}
    print(f"\n{'couche':>6} | {'cos(perm)':>10} | {'cos(autre)':>11} | {'invariance':>11}")
    print("-" * 48)
    for layer in range(n_layers):
        w = float(within[:, layer].mean())
        a = float(across[:, layer].mean())
        score = (w - a) / (1 - a + 1e-9)   
        res["layers"].append(
            {"layer": layer, "cos_perm": w, "cos_other": a, "invariance": score}
        )
        print(f"{layer:>6} | {w:>10.3f} | {a:>11.3f} | {score:>11.3f}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nÉcrit dans {args.out}")

    _plot(res, args.plot)


if __name__ == "__main__":
    main()
