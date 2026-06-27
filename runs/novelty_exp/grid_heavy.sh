#!/usr/bin/env bash
# HEAVY-evolution test: does ESN evolution find the last 1% (2.61->2.635) that
# best-of-N can't? Matched ~160-eval budget. single=best-of-160 (sampling),
# iter_nov=40 gens x batch4 (heavy evolution). scipy seed, 90s, OE prompt, nz.
set -u
WT=/Users/nadavbarkai/dev/esn/.claude/worktrees/novelty-experiments; cd "$WT"
PY=/Users/nadavbarkai/dev/esn/.venv/bin/python
OUT="$WT/runs/novelty_exp/results_heavy.jsonl"; : > "$OUT"
LOG="$WT/runs/novelty_exp/heavy.log"; : > "$LOG"
export PYTHONPATH="$WT/src:$WT/examples:$WT/runs/h2h_bf:$WT/runs/novelty_exp"
export OPENAI_API_KEY="${OPENAI_API_KEY_ESN:-$OPENAI_API_KEY}"
export NEUTRALIZE_GATE=1 DOMAIN=nz GEN_MODEL=gpt-4o-mini OPENEVOLVE_PROMPT=1
export NZ_TIMEOUT=90 NZ_SEED="$WT/runs/h2h_bf/scipy_seed.py"
run(){ echo "=== $(date +%H:%M:%S) $5 seed=$4 ===" | tee -a "$LOG"
  "$PY" "$WT/runs/novelty_exp/run_specdim.py" "$1" "$4" "$2" "$3" 2>>"$LOG" \
    | grep '^SPECDIM_RESULT ' | sed 's/^SPECDIM_RESULT //' \
    | $PY -c "import sys,json; d=json.loads(sys.stdin.read()); d['label']='$5'; print(json.dumps(d))" | tee -a "$OUT"; }
for seed in 42 43; do
  run off 1  160 "$seed" bestof160
  run 8   40 4   "$seed" iter_nov40
done
echo "=== HEAVY DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"
