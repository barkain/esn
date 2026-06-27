#!/usr/bin/env bash
# CLEAN comparison: gate NEUTRALIZED (upstream-equivalent engine), identical
# prompt (ESN mutator), 4 seeds. Arms:
#   bestofN : novelty-off, gens=2, batch=80  (80 seed-mutations, no iteration)
#   esn_off : novelty-off, gens=20, batch=4  (~80 evals, iteration)
#   esn_on8 : novelty-on spectral_dim=8, gens=20, batch=4
set -u
WT=/Users/nadavbarkai/dev/esn/.claude/worktrees/novelty-experiments; cd "$WT"
PY=/Users/nadavbarkai/dev/esn/.venv/bin/python
OUT="$WT/runs/novelty_exp/results_clean.jsonl"; : > "$OUT"
LOG="$WT/runs/novelty_exp/grid_clean.log"; : > "$LOG"
export PYTHONPATH="$WT/src:$WT/examples"
export OPENAI_API_KEY="${OPENAI_API_KEY_ESN:-$OPENAI_API_KEY}"
export NEUTRALIZE_GATE=1
run(){ # arm gens batch seed label
  echo "=== $(date +%H:%M:%S) $5 seed=$4 ===" | tee -a "$LOG"
  "$PY" "$WT/runs/novelty_exp/run_specdim.py" "$1" "$4" "$2" "$3" 2>>"$LOG" \
    | grep '^SPECDIM_RESULT ' | sed 's/^SPECDIM_RESULT //' \
    | $PY -c "import sys,json; d=json.loads(sys.stdin.read()); d['label']='$5'; print(json.dumps(d))" | tee -a "$OUT"
}
for seed in 42 43 44 45; do
  run off 2  80 "$seed" bestofN
  run off 20 4  "$seed" esn_off
  run 8   20 4  "$seed" esn_on8
done
echo "=== CLEAN GRID DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"
