#!/usr/bin/env bash
# Regenerate paper figures.
#
# Assumes:
#   - activation companion arrays have already been collected under results/
#
# If cached plot data exists under plots/data/, scripts use it where supported.
# If it is missing, scripts recompute it. The probe-cosine and neuron-boxplot
# commands always refit/read activations because their scripts do not currently
# have a pure plotting cache path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run() {
  printf '\n>>> %s\n' "$*"
  "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

run_with_optional_data() {
  local script="$1"
  local data="$2"
  shift 2
  if [[ -f "$data" ]]; then
    run uv run python "$script" --data "$data" "$@"
  else
    run uv run python "$script" "$@"
  fi
}

mkdir -p plots/figures

run_with_optional_data \
  plots/scripts/fig_ablation_structure_present_patch.py \
  plots/data/fig_ablation_structure_present_patch.csv

run_with_optional_data \
  plots/scripts/fig_head_attention_cells.py \
  plots/data/fig_head_attention_cells.npz

run_with_optional_data \
  plots/scripts/fig_probe_acc_by_layer.py \
  plots/data/fig_probe_acc_by_layer.csv

run_with_optional_data \
  plots/scripts/fig_probe_auc_by_layer.py \
  plots/data/fig_probe_auc_by_layer.csv

run_with_optional_data \
  plots/scripts/fig_structure_probe_layer_transfer.py \
  plots/data/fig_structure_probe_layer_transfer.csv

run_with_optional_data \
  plots/scripts/fig_structure_probe_transfer.py \
  plots/data/fig_structure_probe_transfer.csv

run_with_optional_data \
  plots/scripts/fig_unembed_similarity.py \
  plots/data/fig_unembed_similarity.npy

# Reproduce fig_last_layer_ns_margin.pdf and fig_last_layer_ns_lens.pdf.
if [[ -f plots/data/fig_last_layer_naked_singles.csv ]]; then
  run uv run python plots/scripts/fig_last_layer_naked_singles.py \
    --data plots/data/fig_last_layer_naked_singles.csv \
    --puzzle 0 \
    --pos 25
else
  run uv run python plots/scripts/fig_last_layer_naked_singles.py
fi

# fig_head_circuit.py needs precomputed attention and DLA data. Generate those
# inputs first if they are not already present.
if [[ ! -f plots/data/fig_head_attention_cells.npz ]]; then
  run uv run python plots/scripts/fig_head_attention_cells.py
fi

if [[ ! -f plots/data/fig_head_dla_substructure.npz ]]; then
  run uv run python plots/scripts/fig_head_dla_substructure.py \
    --layer 4 \
    --head 6 \
    --sub cols \
    --instance 4
fi

require_file plots/data/fig_head_attention_cells.npz
require_file plots/data/fig_head_dla_substructure.npz

# Circuit panels. 
run uv run python plots/scripts/fig_head_circuit.py \
  --attn-data plots/data/fig_head_attention_cells.npz \
  --dla-data plots/data/fig_head_dla_substructure.npz \
  --layer 4 \
  --head 6 \
  --sub cols \
  --instance 4

run uv run python plots/scripts/fig_head_circuit.py \
  --attn-data plots/data/fig_head_attention_cells.npz \
  --dla-data plots/data/fig_head_dla_substructure.npz \
  --layer 5 \
  --head 8 \
  --sub rows \
  --instance 7

run uv run python plots/scripts/fig_head_circuit.py \
  --attn-data plots/data/fig_head_attention_cells.npz \
  --dla-data plots/data/fig_head_dla_substructure.npz \
  --layer 6 \
  --head 3 \
  --sub boxes \
  --instance 5

# Expensive regeneration commands.

run uv run python plots/scripts/fig_neuron_candidate_sensitivity.py \
  --boxplot 39 1315

run uv run python plots/scripts/fig_probe_cosine_sim.py \
  --cell 57 \
  --digit 3 \
  --layer 5

cat <<'EOF'

EOF
