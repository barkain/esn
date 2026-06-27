#!/usr/bin/env bash
set -u
WT=/Users/nadavbarkai/dev/esn/.claude/worktrees/novelty-experiments; cd "$WT"
PY=/Users/nadavbarkai/dev/esn/.venv/bin/python
export PYTHONPATH="$WT/src:$WT/examples:$WT/runs/h2h_bf"
export OPENAI_API_KEY="${OPENAI_API_KEY_ESN:-$OPENAI_API_KEY}"
OUT="$WT/runs/novelty_exp/results_noantiopt.jsonl"; : > "$OUT"
for seed in 42 43 44; do
  "$PY" runs/novelty_exp/test_noantiopt.py "$seed" 20 2>>"$WT/runs/novelty_exp/noantiopt.log" | grep '^NOANTIOPT' | tee -a "$OUT"
done
echo DONE
