"""Étape 4 — test d'invariance par permutation des indices.

Sur le checkpoint final (le modèle est censé avoir une représentation
structurelle du Sudoku), on permute l'ordre des tokens d'indices (avant SEP)
et on mesure, couche par couche, si le residual stream à la position SEP reste
identique. Une représentation invariante à l'ordre des indices doit donner des
activations ~identiques sous permutation.

Usage (Colab, après extraction du checkpoint des artefacts) :
    uv run python scripts/exp_clue_permutation.py \
        --ckpt results/3M-backtracking-packing/checkpoint \
        --traces bt_traces_3m.npz --n 256 --k 8
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/3M-backtracking-packing/checkpoint")
    ap.add_argument("--traces", default="bt_traces_3m.npz")
    ap.add_argument("--n", type=int, default=256, help="nombre de grilles")
    ap.add_argument("--k", type=int, default=8, help="permutations par grille")
    ap.add_argument("--chunk", type=int, default=32, help="grilles par chunk (mémoire)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="clue_perm_invariance.json")
    args = ap.parse_args()

    random.seed(0)

    params, model = load_checkpoint(args.ckpt)
    fn = make_intermediates_fn(model)

    tp = np.load(args.traces, allow_pickle=True)
    pkey = "puzzles_test" if "puzzles_test" in tp else "puzzles_val"
    puzzles = [_as_str(p) for p in tp[pkey][: args.n]]
    print(f"{len(puzzles)} grilles, {args.k} permutations chacune", flush=True)

    # On ne garde que le vecteur à la position SEP (mémoire).
    sep_canon: list[tuple[int, np.ndarray]] = []   # (idx grille, (n_layers, d))
    sep_perm: list[np.ndarray] = []                # (n_layers, d) par permutation
    perm_owner: list[int] = []                     # idx grille de chaque permutation

    for c0 in range(0, len(puzzles), args.chunk):
        chunk = puzzles[c0 : c0 + args.chunk]
        seqs, tag = [], []   # tag = ("canon"|"perm", idx local)
        for li, p in enumerate(chunk):
            seqs.append(encode_clues(p, randomize_order=False, use_sep=True))
            tag.append(("canon", li))
            for _ in range(args.k):
                seqs.append(encode_clues(p, randomize_order=True, use_sep=True))
                tag.append(("perm", li))
        acts = collect_activations(fn, params, seqs, args.batch)["post_mlp"]  # (M, L, T, d)
        for m, (kind, li) in enumerate(tag):
            sep = len(seqs[m]) - 1            # position du token SEP
            v = acts[m, :, sep, :]            # (n_layers, d)
            if kind == "canon":
                sep_canon.append((c0 + li, v))
            else:
                sep_perm.append(v)
                perm_owner.append(c0 + li)
        del acts

    n_layers = sep_canon[0][1].shape[0]
    canon_by_idx = {gi: v for gi, v in sep_canon}

    # within-puzzle : cos(canonique, permutation) par couche
    within = np.stack(
        [cos(canon_by_idx[gi], v) for gi, v in zip(perm_owner, sep_perm)], 0
    )  # (n_perm, n_layers)

    # across-puzzle : cos(canonique_i, canonique_j) sur des paires aléatoires
    canon_arr = np.stack([v for _, v in sep_canon], 0)   # (n, n_layers, d)
    rng = np.random.default_rng(0)
    pairs = np.stack(
        [rng.choice(len(canon_arr), 2, replace=False) for _ in range(4000)], 0
    )
    across = cos(canon_arr[pairs[:, 0]], canon_arr[pairs[:, 1]])   # (4000, n_layers)

    res = {"n_layers": int(n_layers), "n_puzzles": len(puzzles), "k": args.k, "layers": []}
    print(f"\n{'couche':>6} | {'cos(perm)':>10} | {'cos(autre)':>11} | {'invariance':>11}")
    print("-" * 48)
    for layer in range(n_layers):
        w = float(within[:, layer].mean())
        a = float(across[:, layer].mean())
        score = (w - a) / (1 - a + 1e-9)   # 1 = invariant, 0 = pas mieux qu'aléatoire
        res["layers"].append(
            {"layer": layer, "cos_perm": w, "cos_other": a, "invariance": score}
        )
        print(f"{layer:>6} | {w:>10.3f} | {a:>11.3f} | {score:>11.3f}")

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nÉcrit dans {args.out}")


if __name__ == "__main__":
    main()
