#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Controlled verification that FIFO / LRU / RL evict correctly. Run IN THE VM.
#
# Uses a small flow table (FLOWRL_MAX_FLOWS) and a deterministic, distinct-pair
# workload so eviction triggers early and the victim at each step is predictable.
# Captures the controller stdout (where the "removing ..." lines are printed) and
# checks it with tests/parse_eviction_log.py.
#
#   bash scripts/verify_controllers.sh fifo        # one controller
#   for c in fifo lru rl; do bash scripts/verify_controllers.sh $c; done   # all
#
# Args:  $1 controller (fifo|lru|rl, default fifo)
#        $2 table capacity      (default 5)
#        $3 number of hosts     (default 8  → 56 distinct pairs)
# ─────────────────────────────────────────────────────────────────────────────
set -e

CTRL="${1:-fifo}"
MAXF="${2:-5}"
HOSTS="${3:-8}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="verify_${CTRL}_${STAMP}.log"

echo "[1/4] Generating deterministic workload (${HOSTS} hosts, distinct pairs)..."
python3 tests/generate_data.py --hosts "$HOSTS" --sequential

echo "[2/4] Starting ${CTRL} controller with FLOWRL_MAX_FLOWS=${MAXF} (stdout -> ${OUT})..."
FLOWRL_MAX_FLOWS="$MAXF" PYTHONPATH="$PWD" \
    ryu-manager "controllers/${CTRL}_controller.py" > "$OUT" 2>&1 &
RYU_PID=$!
sleep 5   # let the controller connect on 6633

echo "[3/4] Running Mininet topology (feeding 'exit' so the CLI does not block)..."
echo exit | sudo python3 tests/topology.py --controller "$CTRL" --hosts "$HOSTS" --runs 1 \
    || true
sleep 2
kill "$RYU_PID" 2>/dev/null || true
wait "$RYU_PID" 2>/dev/null || true

echo "[4/4] Analysing eviction log..."
python3 tests/parse_eviction_log.py "$OUT" --controller "$CTRL" --max-flows "$MAXF"

echo
echo "Raw controller stdout kept at: ${OUT}"
echo "Eviction lines:"
grep -E "removing (oldest|LRU|entry)" "$OUT" | head -10 || echo "  (none found — eviction never fired!)"
