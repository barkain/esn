#!/usr/bin/env bash
# Jackpot-rate test: does ITERATION reach the 2.5 grid more often than single-shot
# best-of-N? Both novelty-off, gate off. 8 seeds. Count best>=2.4 per arm.
set -u
WT=/Users/nadavbarkai/dev/esn/.claude/worktrees/novelty-experiments; cd "$WT"
PY=/Users/nadavbarkai/dev/esn/.venv/bin/python
OUT="$WT/runs/novelty_exp/results_jackpot.jsonl"; : > "$OUT"
LOG="$WT/runs/novelty_exp/jackpot.log"; : > "$LOG"
export PYTHONPATH="$WT/src:$WT/examples"
export OPENAI_API_KEY="${OPENAI_API_KEY_ESN:-$OPENAI_API_KEY}"
export NEUTRALIZE_GATE=1
run(){ # arm gens batch seed label
  echo "=== $(date +%H:%M:%S) $5 seed=$4 ===" | tee -a "$LOG"
  "$PY" "$WT/runs/novelty_exp/run_specdim.py" "$1" "$4" "$2" "$3" 2>>"$LOG" \
    | grep '^SPECDIM_RESULT ' | sed 's/^SPECDIM_RESULT //' \
    | $PY -c "import sys,json; d=json.loads(sys.stdin.read()); d['label']='$5'; print(json.dumps(d))" | tee -a "$OUT"
}
for seed in 50 51 52 53 54 55 56 57; do
  run off 1  80 "$seed" single
  run off 20 4  "$seed" iter
done
echo "=== JACKPOT DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"
