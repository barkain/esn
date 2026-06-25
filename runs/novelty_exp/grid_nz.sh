#!/usr/bin/env bash
# MEANINGFUL test on the de-degenerated task (DOMAIN=nz, r>0 enforced).
# Does ESN iteration push 4o-mini ABOVE the 2.167 uniform-grid plateau where
# single-shot can't? Arms: single (gens=1 b=80), iter (gens=20 b=4),
# iter_nov (novelty spectral_dim=8, gens=20 b=4). Clean engine. 8 seeds.
set -u
WT=/Users/nadavbarkai/dev/esn/.claude/worktrees/novelty-experiments; cd "$WT"
PY=/Users/nadavbarkai/dev/esn/.venv/bin/python
OUT="$WT/runs/novelty_exp/results_nz.jsonl"; : > "$OUT"
LOG="$WT/runs/novelty_exp/nz.log"; : > "$LOG"
export PYTHONPATH="$WT/src:$WT/examples:$WT/runs/h2h_bf"
export OPENAI_API_KEY="${OPENAI_API_KEY_ESN:-$OPENAI_API_KEY}"
export NEUTRALIZE_GATE=1 DOMAIN=nz NZ_MIN_RADIUS=0.0 GEN_MODEL=gpt-4o-mini
run(){ # arm gens batch seed label
  echo "=== $(date +%H:%M:%S) $5 seed=$4 ===" | tee -a "$LOG"
  "$PY" "$WT/runs/novelty_exp/run_specdim.py" "$1" "$4" "$2" "$3" 2>>"$LOG" \
    | grep '^SPECDIM_RESULT ' | sed 's/^SPECDIM_RESULT //' \
    | $PY -c "import sys,json; d=json.loads(sys.stdin.read()); d['label']='$5'; print(json.dumps(d))" | tee -a "$OUT"
}
for seed in 42 43 44 45 46 47 48 49; do
  run off 1  80 "$seed" single
  run off 20 4  "$seed" iter
  run 8   20 4  "$seed" iter_nov
done
echo "=== NZ GRID DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"
