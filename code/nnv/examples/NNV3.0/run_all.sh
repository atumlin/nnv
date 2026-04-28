#!/usr/bin/env bash
# Run the full NNV3.0 repeatability suite (FairNNV + ProbVer + GNNV + VideoStar)
# in one shot. Each experiment runs in its own MATLAB session so a crash in
# one does not lose the others.
#
# Usage (inside the nnv3.0 container, or any MATLAB+NNV install):
#   bash code/nnv/examples/NNV3.0/run_all.sh
#
# Optional environment overrides:
#   NNV3_SKIP="probver videostar"     # space-separated experiments to skip
#   NNV3_LOG_DIR=/tmp/nnv3_logs       # where per-experiment stdout lands
#   MATLAB=/usr/local/bin/matlab      # MATLAB binary (default: 'matlab' on PATH)
#
# Exit code is 0 only if every selected experiment finished cleanly. The final
# summary table (and a CSV at $NNV3_LOG_DIR/summary.csv) lists wall-clock time
# per experiment.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNV_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"  # .../code/nnv
LOG_DIR="${NNV3_LOG_DIR:-${SCRIPT_DIR}/repeatability_logs}"
MATLAB="${MATLAB:-matlab}"
SKIP="${NNV3_SKIP:-}"

mkdir -p "$LOG_DIR"
SUMMARY_CSV="${LOG_DIR}/summary.csv"
echo "experiment,status,wall_seconds,log" > "$SUMMARY_CSV"

# Forward-compat call is repeated in each script too; we set it here as well
# so MATLAB doesn't error on Blackwell/RTX 5090 hosts before the script runs.
MATLAB_PRELUDE="addpath(genpath('${NNV_ROOT}')); try, parallel.gpu.enableCUDAForwardCompatibility(true); catch; end;"

run_one() {
    local name="$1"
    local subdir="$2"
    local script="$3"

    if [[ " $SKIP " == *" $name "* ]]; then
        printf '\n=== %-12s SKIPPED (NNV3_SKIP) ===\n' "$name"
        echo "${name},skipped,0," >> "$SUMMARY_CSV"
        return 0
    fi

    local logfile="${LOG_DIR}/${name}.log"
    printf '\n=== %-12s start: %s ===\n' "$name" "$(date -u +%FT%TZ)"
    local t0
    t0=$(date -u +%s)
    (
        cd "${SCRIPT_DIR}/${subdir}" || exit 99
        "$MATLAB" -nodisplay -batch "${MATLAB_PRELUDE} run('${script}'); exit()"
    ) 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    local t1
    t1=$(date -u +%s)
    local elapsed=$(( t1 - t0 ))
    if [[ $rc -eq 0 ]]; then
        printf '=== %-12s OK in %ds (log: %s) ===\n' "$name" "$elapsed" "$logfile"
        echo "${name},ok,${elapsed},${logfile}" >> "$SUMMARY_CSV"
    else
        printf '=== %-12s FAILED (rc=%d) in %ds (log: %s) ===\n' "$name" "$rc" "$elapsed" "$logfile"
        echo "${name},failed,${elapsed},${logfile}" >> "$SUMMARY_CSV"
    fi
    return $rc
}

overall=0
run_one fairnnv  FairNNV  run_fm26_fairnnv.m   || overall=$?
run_one probver  ProbVer  run_probver.m        || overall=$?
run_one gnnv     GNNV     run_gnn_experiments.m || overall=$?
run_one videostar VideoStar run_zoomin_4f.m    || overall=$?

echo
echo "=========================== SUMMARY ============================"
column -t -s, "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
echo "================================================================"
echo "Per-experiment logs: $LOG_DIR"
exit "$overall"
