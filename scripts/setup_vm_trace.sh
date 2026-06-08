#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Collect a REAL flow-table trace inside the Ubuntu VM (Mininet already installed).
#
# Uses the FIFO controller, which needs only Ryu + colorlog (no torch), so it is
# the lightest way to capture realistic flow lifetimes for the delayed-reward
# simulator. Run from the repo root inside the VM:
#
#     git clone https://github.com/iwa147369/flow-tabale-management-using-rl.git
#     cd flow-tabale-management-using-rl
#     bash scripts/setup_vm_trace.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "[1/3] Installing controller dependencies (Ryu + colorlog)..."
# Ryu is picky on Python 3.10+. If this fails, see the NOTE at the bottom.
python3 -m pip install --user ryu colorlog

echo "[2/3] Generating the 20-host traffic file (20_hosts_test.txt)..."
python3 tests/generate_data.py

echo "[3/3] Setup complete."
cat <<'EOF'

────────────────────────────────────────────────────────────────────────────
COLLECT THE TRACE — use TWO terminals in the VM, both from the repo root:

  # Terminal A — start the recording FIFO controller (note PYTHONPATH=$PWD)
  FLOWRL_RECORD_TRACE=1 PYTHONPATH="$PWD" ryu-manager controllers/fifo_controller.py

  # Terminal B — run the Mininet experiment (defaults to controller 127.0.0.1:6633)
  sudo python3 tests/topology.py --controller fifo --runs 1
  #   When the Mininet CLI prompt appears at the end, type:  exit

  # Back in Terminal A, stop the controller with Ctrl+C.
  # The trace auto-saves on shutdown:
  ls -la traces/*.pkl

COPY THE TRACE BACK TO THE HOST (run on the host, or scp from the VM):
  mkdir -p traces/collected/fifo
  scp <vmuser>@192.168.122.85:~/flow-tabale-management-using-rl/traces/trace_fifo_*.pkl \
      traces/collected/fifo/

THEN ON THE HOST — train the pointer/PPO agent on the real trace:
  .venv/bin/python -m training.train_ppo --episodes 300 \
      --trace-file traces/collected/fifo/trace_fifo_<date>.pkl
────────────────────────────────────────────────────────────────────────────

NOTE — if "pip install ryu" failed on the VM's Python, pin the known-good deps:
  python3 -m pip install --user 'eventlet==0.30.2' 'dnspython==1.16.0' ryu colorlog
Also ensure ~/.local/bin is on PATH so the 'ryu-manager' command is found:
  export PATH="$HOME/.local/bin:$PATH"
EOF
