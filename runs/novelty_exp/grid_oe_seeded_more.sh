#!/usr/bin/env bash
# Powered extension of the fair ESN test: 8 more seeds (45-52). Same setup —
# scipy seed ~2.58, 90s timeout, OpenEvolve prompt, nz. Appends to a new file;
# combine with results_oe_seeded.jsonl (seeds 42-44) for n=11.
set -u
WT=/Users/nadavbarkai/dev/esn/.claude/worktrees/novelty-experiments; cd "$WT"
PY=/Users/nadavbarkai/dev/esn/.venv/bin/python
OUT="$WT/runs/novelty_exp/results_oe_seeded_ext.jsonl"; : > "$OUT"
LOG="$WT/runs/novelty_exp/oe_seeded_ext.log"; : > "$LOG"
export PYTHONPATH="$WT/src:$WT/examples:$WT/runs/h2h_bf:$WT/runs/novelty_exp"
export OPENAI_API_KEY="${OPENAI_API_KEY_ESN:-$OPENAI_API_KEY}"
export NEUTRALIZE_GATE=1 DOMAIN=nz GEN_MODEL=gpt-4o-mini OPENEVOLVE_PROMPT=1
export NZ_TIMEOUT=90 NZ_SEED="$WT/runs/h2h_bf/scipy_seed.py"
run(){
  echo "=== $(date +%H:%M:%S) $5 seed=$4 ===" | tee -a "$LOG"
  "$PY" "$WT/runs/novelty_exp/run_specdim.py" "$1" "$4" "$2" "$3" 2>>"$LOG" \
    | grep '^SPECDIM_RESULT ' | sed 's/^SPECDIM_RESULT //' \
    | $PY -c "import sys,json; d=json.loads(sys.stdin.read()); d['label']='$5'; print(json.dumps(d))" | tee -a "$OUT"
}
for seed in 45 46 47 48 49 50 51 52; do
  run off 1  80 "$seed" single
  run off 20 4  "$seed" iter
  run 8   20 4  "$seed" iter_nov
done
echo "=== EXT DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"
