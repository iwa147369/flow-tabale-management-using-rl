#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Collect FIFO, LRU and RL traces sequentially on ONE shared traffic pattern.
# Run IN THE VM (needs Mininet + Ryu). This is the fair-comparison collector: the
# same 20_hosts_test.txt drives all three controllers, so the resulting traces
# differ only by eviction policy.
#
#   bash scripts/collect_all_traces.sh            # 20 hosts, 1 run each
#   bash scripts/collect_all_traces.sh 20 3       # 20 hosts, 3 runs each
#
# Args:  $1 hosts (default 20)   $2 runs per controller (default 1)
#
# Prereqs (one-time): bash scripts/setup_vm_trace.sh  (Ryu + colorlog, pinned
# eventlet) and, for RL, torch + numpy installed (see docs/memory.md §6).
# ─────────────────────────────────────────────────────────────────────────────
set -u

HOSTS="${1:-20}"
RUNS="${2:-1}"
TRAFFIC="${HOSTS}_hosts_test.txt"

echo "[setup] Generating ONE shared traffic file: ${TRAFFIC}"
python3 tests/generate_data.py --hosts "$HOSTS"

for c in fifo lru rl; do
    echo
    echo "============================================================"
    echo " Collecting trace: ${c}  (hosts=${HOSTS}, runs=${RUNS})"
    echo "============================================================"

    pkill -f ryu-manager 2>/dev/null || true
    sleep 2   # let port 6633 free up

    out="ryu_${c}.out"
    FLOWRL_RECORD_TRACE=1 PYTHONPATH="$PWD" \
        ryu-manager "controllers/${c}_controller.py" > "$out" 2>&1 &
    ryu_pid=$!
    sleep 5

    if ! kill -0 "$ryu_pid" 2>/dev/null; then
        echo "ERROR: ${c} controller died at startup. Last lines of ${out}:" >&2
        tail -15 "$out" >&2
        echo "Hint: eventlet ImportError? pin: pip install --user 'eventlet==0.30.2' 'dnspython==1.16.0'" >&2
        exit 1
    fi

    # Run the experiment; feed 'exit' so the Mininet CLI does not block.
    echo exit | sudo python3 tests/topology.py --controller "$c" --hosts "$HOSTS" --runs "$RUNS" || true
    sleep 2

    # SIGINT (not SIGTERM) so Ryu shuts down cleanly and the controller's
    # __del__ auto-saves the trace to traces/trace_<c>_<timestamp>.pkl.
    kill -INT "$ryu_pid" 2>/dev/null || true
    wait "$ryu_pid" 2>/dev/null || true
    sleep 1

    mkdir -p "traces/collected/${c}"
    latest="$(ls -t traces/trace_${c}_*.pkl 2>/dev/null | head -1)"
    if [ -n "$latest" ]; then
        mv "$latest" "traces/collected/${c}/"
        echo "  -> saved traces/collected/${c}/$(basename "$latest")"
    else
        echo "  WARN: no trace file produced for ${c} (check ${out})"
    fi
done

echo
echo "Done. Collected traces:"
ls -la traces/collected/*/ 2>/dev/null
echo
echo "Copy them to the host to train, e.g.:"
echo "  scp 'iwa@192.168.122.85:$PWD/traces/collected/fifo/*.pkl' traces/collected/fifo/"
