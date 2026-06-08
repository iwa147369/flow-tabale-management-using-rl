"""
Helper script to make collecting real traces from Mininet easier.

Usage:
    1. Start Ryu with recording enabled:
       FLOWRL_RECORD_TRACE=1 ryu-manager controllers/rl_controller.py

    2. In another terminal, run your experiment with this script (or manually).

    3. After the experiment finishes, this script can help you save the trace
       if you didn't add auto-save logic.

This is a convenience wrapper. The real recording happens inside the Ryu controller.
"""

import os
import sys
from pathlib import Path


def print_instructions():
    print("=" * 70)
    print("REAL TRACE COLLECTION WORKFLOW")
    print("=" * 70)
    print()
    print("1. Enable trace recording:")
    print("   export FLOWRL_RECORD_TRACE=1")
    print()
    print("2. Start your desired controller:")
    print("   ryu-manager controllers/rl_controller.py")
    print("   # or fifo_controller.py / lru_controller.py")
    print()
    print("3. In another terminal, run your Mininet experiment:")
    print("   python tests/topology.py --controller rl --runs 1")
    print("   (or your own experiment script)")
    print()
    print("4. Traces should be auto-saved to ./traces/ when the controller shuts down.")
    print("   If not, you can manually call save_trace() from the Ryu shell or")
    print("   add it at the end of your experiment.")
    print()
    print("5. After collection, use the trace like this:")
    print()
    print("   from training.trace_simulator import TraceRecorder, TraceSimulator")
    print("   from training.environment import FlowTableEnvironment")
    print()
    print("   rec = TraceRecorder.load('traces/trace_rl_....pkl')")
    print("   sim = TraceSimulator(rec.get_trace())")
    print("   env = FlowTableEnvironment(trace_simulator=sim)")
    print()
    print("=" * 70)
    print("Recommended: Collect at least 3 traces per controller (FIFO, LRU, RL)")
    print("under the same traffic conditions for fair comparison.")
    print("=" * 70)


if __name__ == "__main__":
    print_instructions()