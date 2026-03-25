#!/usr/bin/env bash
set -euo pipefail
# run_hugine_suite.sh
# Full HuGINE verification experiments: PF + OPF + edge perturbation.
# Single MATLAB session, all serial for fair timing.
# Launch with: nohup ./run_hugine_suite.sh > /dev/null 2>&1 &

# ── Configure ────────────────────────────────────────────────────────────
NTFY_TOPIC="gnnv-$(hostname -s)-hugine"
NNV_ROOT="/home/verivital/Anne/dev/gnnv2/nnv/code/nnv"
SAIV26_DIR="$NNV_ROOT/examples/NN/GNN/SAIV26"
MATLAB="/usr/local/MATLAB/R2024a/bin/matlab"
CURL="/home/verivital/anaconda3/bin/curl"
GNN_TRAIN="/home/verivital/Anne/dev/gnnv2/gnn_training/outputs"
LOG_DIR="$SAIV26_DIR/results/logs"
mkdir -p "$LOG_DIR"
SUITE_LOG="$LOG_DIR/hugine_suite_$(date +%Y%m%d-%H%M%S).log"

notify() {
    local title="$1" msg="$2" priority="${3:-default}"
    "$CURL" -s \
        -H "Title: $title" \
        -H "Priority: $priority" \
        -H "Tags: zap" \
        -d "$msg" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    notify-send "$title" "$msg" 2>/dev/null || true
}

echo "============================================" | tee "$SUITE_LOG"
echo "  HuGINE Full Experiment Suite (SERIAL)"     | tee -a "$SUITE_LOG"
echo "  Log:  $SUITE_LOG"                          | tee -a "$SUITE_LOG"
echo "  ntfy: https://ntfy.sh/$NTFY_TOPIC"         | tee -a "$SUITE_LOG"
echo "  Started: $(date)"                          | tee -a "$SUITE_LOG"
echo "============================================" | tee -a "$SUITE_LOG"

notify "HuGINE Suite Started" "Serial suite started at $(date +%H:%M:%S). Single MATLAB session: PF → OPF → Edge → Figures."

cd "$NNV_ROOT"

# ── Step 0: Copy best-seed model files ───────────────────────────────────
echo "--- Step 0: Deploy model files --- $(date +%H:%M:%S)" | tee -a "$SUITE_LOG"

copy_if_missing() {
    local src="$1" dst="$2"
    if [ ! -f "$dst" ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        echo "  Copied: $(basename "$dst")" | tee -a "$SUITE_LOG"
    fi
}

# Best seeds: ieee24_pf=4, ieee39_pf=4, ieee118_pf=3, ieee24_opf=4, ieee39_opf=3, ieee118_opf=4
copy_if_missing "$GNN_TRAIN/ieee24_pf/seed_4/gine_pretrain_pf_ieee24.mat" \
                "$SAIV26_DIR/PowerFlow/IEEE24/models/gine_pretrain_pf_ieee24.mat"
copy_if_missing "$GNN_TRAIN/ieee39_pf/seed_4/gine_pretrain_pf_ieee39.mat" \
                "$SAIV26_DIR/PowerFlow/IEEE39/models/gine_pretrain_pf_ieee39.mat"
copy_if_missing "$GNN_TRAIN/ieee118_pf/seed_3/gine_pretrain_pf_ieee118.mat" \
                "$SAIV26_DIR/PowerFlow/IEEE118/models/gine_pretrain_pf_ieee118.mat"
copy_if_missing "$GNN_TRAIN/ieee24_opf/seed_4/gine_pretrain_opf_ieee24.mat" \
                "$SAIV26_DIR/OptimalPowerFlow/IEEE24/models/gine_pretrain_opf_ieee24.mat"
copy_if_missing "$GNN_TRAIN/ieee39_opf/seed_3/gine_pretrain_opf_ieee39.mat" \
                "$SAIV26_DIR/OptimalPowerFlow/IEEE39/models/gine_pretrain_opf_ieee39.mat"
copy_if_missing "$GNN_TRAIN/ieee118_opf/seed_4/gine_pretrain_opf_ieee118.mat" \
                "$SAIV26_DIR/OptimalPowerFlow/IEEE118/models/gine_pretrain_opf_ieee118.mat"

echo "  Models deployed." | tee -a "$SUITE_LOG"

# ── Run all experiments in a single MATLAB session ───────────────────────
echo "--- Starting MATLAB (single session) --- $(date +%H:%M:%S)" | tee -a "$SUITE_LOG"
notify "MATLAB Starting" "Single session for all experiments at $(date +%H:%M:%S)."

"$MATLAB" -nodisplay -nosplash \
    -r "addpath(genpath('.')); run_hugine_full_experiments('parallel', false); exit" \
    2>&1 | tee "$LOG_DIR/hugine_full_$(date +%Y%m%d-%H%M%S).log"
EC=$?

if [ "$EC" -ne 0 ]; then
    echo "FAILED: MATLAB experiments (exit $EC) $(date +%H:%M:%S)" | tee -a "$SUITE_LOG"
    notify "HuGINE FAILED" "MATLAB experiments failed (exit $EC) at $(date +%H:%M:%S)" "high"
else
    echo "OK: MATLAB experiments $(date +%H:%M:%S)" | tee -a "$SUITE_LOG"
    notify "HuGINE Experiments Done" "All experiments completed at $(date +%H:%M:%S). Generating figures..."
fi

# ── Generate figures and LaTeX tables ────────────────────────────────────
echo "--- Generating figures + LaTeX --- $(date +%H:%M:%S)" | tee -a "$SUITE_LOG"

LATEST_RESULTS=$(ls -td "$SAIV26_DIR"/results/hugine_* 2>/dev/null | head -1)
if [ -n "$LATEST_RESULTS" ] && [ -f "$LATEST_RESULTS/hugine_results.csv" ]; then
    conda run -n at_ml python3 "$SAIV26_DIR/generate_hugine_figures.py" \
        "$LATEST_RESULTS/hugine_results.csv" \
        -o "$LATEST_RESULTS/figures" \
        --latex 2>&1 | tee -a "$SUITE_LOG"
    FIG_EC=$?
    if [ "$FIG_EC" -ne 0 ]; then
        echo "WARNING: Figure generation failed (exit $FIG_EC)" | tee -a "$SUITE_LOG"
        notify "Figures Failed" "Figure generation failed (exit $FIG_EC)" "high"
    else
        echo "OK: Figures generated in $LATEST_RESULTS/figures/" | tee -a "$SUITE_LOG"
    fi
else
    echo "WARNING: No result CSV found, skipping figures" | tee -a "$SUITE_LOG"
fi

echo "" | tee -a "$SUITE_LOG"
echo "=== HuGINE Suite Complete === $(date)" | tee -a "$SUITE_LOG"
notify "HuGINE Suite Complete" "All steps finished at $(date +%H:%M:%S). Results in $SAIV26_DIR/results/." "default"
