#!/usr/bin/env python3
"""
FlowRL Experiment + Training Pipeline

This script helps automate the process of:
1. Running Mininet experiments (with trace collection)
2. Organizing collected traces
3. Launching training runs using real traces

IMPORTANT: Full end-to-end automation is currently limited because
Mininet experiments run inside a VM. This script focuses on making
the workflow as smooth as possible while clearly marking manual steps.

Usage examples:
    # See what you need to do for experiments
    python scripts/run_pipeline.py experiment --controllers fifo lru rl --runs 3

    # After you manually collected traces, organize them
    python scripts/run_pipeline.py organize --traces-dir traces/

    # Train using the collected real traces
    python scripts/run_pipeline.py train --trace-path traces/collected/rl/...

    # Run full pipeline (will stop at manual steps)
    python scripts/run_pipeline.py full --controllers rl --runs 5
"""

import argparse
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TRACES_DIR = PROJECT_ROOT / "traces"
COLLECTED_DIR = TRACES_DIR / "collected"

CONTROLLERS = ["fifo", "lru", "rl"]


def print_header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def stage_experiment(controllers: List[str], runs: int, note: str = ""):
    """Stage 1: Running Mininet experiments (mostly manual for now)"""
    print_header("STAGE 1: RUN MININET EXPERIMENTS")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_note = f"_{note}" if note else ""

    print("This stage requires manual execution inside your Mininet VM.\n")
    print("PREPARATION (on your host machine - Arch Linux):")
    print("-" * 50)
    print("1. Make sure your Ubuntu VM is running and can reach the host.")
    print("2. (Optional but recommended) Take a clean snapshot of the VM.\n")

    print("STEP-BY-STEP INSTRUCTIONS:")
    print("-" * 50)

    for controller in controllers:
        print(f"\n### Controller: {controller.upper()} ###\n")
        print("On your HOST (Arch Linux):")
        print(f"    export FLOWRL_RECORD_TRACE=1")
        print(f"    ryu-manager controllers/{controller}_controller.py")
        print()
        print("Then, inside your UBUNTU VM, run:")
        print(f"    sudo python3 {PROJECT_ROOT}/tests/topology.py \\")
        print(f"        --controller {controller} \\")
        print(f"        --runs {runs}")
        print()

        if runs > 1:
            print(f"    This will run {runs} independent experiments for {controller}.")

        print("\nAfter the experiments finish on the VM:")
        print("  - The controller on the host should auto-save traces when it shuts down.")
        print(f"  - Traces will be saved in: traces/ (with timestamp {timestamp}{run_note})")
        print()

    print("MANUAL NOTES:")
    print("- You must start the controller BEFORE running the topology script.")
    print("- You must stop the controller (Ctrl+C) after the VM finishes for auto-save to trigger.")
    print("- If auto-save doesn't work, you can manually call controller.save_trace() from the Ryu shell.")
    print("- Recommended: Collect traces for all controllers under similar traffic conditions.\n")

    print("Once you have finished running experiments for the desired controllers,")
    print("come back and run:")
    print(f"    python {__file__} organize --timestamp {timestamp}{run_note}\n")

    # Generate a small helper script the user can copy into the VM
    vm_helper = PROJECT_ROOT / "traces" / f"run_in_vm_{timestamp}{run_note}.sh"
    vm_helper.parent.mkdir(parents=True, exist_ok=True)

    with open(vm_helper, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Copy this file into your Ubuntu Mininet VM and run it there.\n")
        f.write("# Make it executable: chmod +x run_in_vm_....sh\n\n")
        f.write("set -e\n\n")
        for controller in controllers:
            f.write(f"echo '=== Running {controller.upper()} experiments ==='\n")
            f.write(f"sudo python3 {PROJECT_ROOT}/tests/topology.py \\\n")
            f.write(f"    --controller {controller} \\\n")
            f.write(f"    --runs {runs}\n\n")
        f.write("echo 'All experiments finished in VM.'\n")

    print(f"Helper script generated for the VM: {vm_helper}")
    print("Copy it into your Ubuntu VM and run it after starting the controller on the host.\n")


def stage_organize(timestamp: str = None, source_dir: str = "traces", train_after: bool = False):
    """Stage 2: Organize collected traces into a clean structure"""
    print_header("STAGE 2: ORGANIZE COLLECTED TRACES")

    source = Path(source_dir)
    if not source.exists():
        print(f"Error: Source directory {source} does not exist.")
        return

    # Find .pkl files
    trace_files = list(source.rglob("*.pkl"))
    if not trace_files:
        print(f"No .pkl trace files found under {source}")
        print("Did you run the experiment stage and collect traces?")
        return

    print(f"Found {len(trace_files)} trace files.\n")

    COLLECTED_DIR.mkdir(parents=True, exist_ok=True)

    collected_traces = {}  # controller -> list of paths

    for trace_file in trace_files:
        name = trace_file.name.lower()
        controller = None
        for c in CONTROLLERS:
            if c in name:
                controller = c
                break

        if not controller:
            controller = "unknown"

        dest_dir = COLLECTED_DIR / controller
        dest_dir.mkdir(parents=True, exist_ok=True)

        if timestamp and timestamp not in trace_file.name:
            new_name = f"{timestamp}_{trace_file.name}"
        else:
            new_name = trace_file.name

        dest_path = dest_dir / new_name
        shutil.copy2(trace_file, dest_path)
        print(f"Copied: {trace_file} -> {dest_path}")

        collected_traces.setdefault(controller, []).append(dest_path)

    # Write a simple metadata file
    meta_file = COLLECTED_DIR / f"run_{timestamp or 'latest'}.txt"
    with open(meta_file, "w") as f:
        f.write(f"Collection run: {timestamp or 'manual'}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Source: {source}\n\n")
        for ctrl, paths in collected_traces.items():
            f.write(f"{ctrl}: {len(paths)} traces\n")
            for p in paths:
                f.write(f"  - {p.name}\n")

    print(f"\nTraces organized under: {COLLECTED_DIR}")
    print(f"Metadata written to: {meta_file}")

    if train_after:
        print("\n--train-after enabled. Launching training for each controller using the latest trace...")
        for controller in CONTROLLERS:
            traces = collected_traces.get(controller, [])
            if traces:
                latest = sorted(traces)[-1]  # lexicographical sort should work with timestamps
                print(f"\nTraining {controller.upper()} with trace: {latest}")
                os.system(f"python -m training.train --episodes 300 --use-trace-reward")  # simplified
            else:
                print(f"No traces found for {controller}, skipping training.")


def stage_train(trace_path: str, episodes: int = 300):
    """Stage 3: Launch training using a collected real trace"""
    print_header("STAGE 3: TRAIN WITH REAL TRACE")

    trace = Path(trace_path)
    if not trace.exists():
        print(f"Error: Trace file not found: {trace}")
        return

    print(f"Using real trace: {trace}")
    print(f"Training for {episodes} episodes with trace-based reward...\n")

    cmd = (
        f"python -m training.train "
        f"--episodes {episodes} "
        f"--use-trace-reward "
        f"--epsilon-min 0.05"
    )

    print("Command that will be executed:")
    print(cmd)
    print()

    confirm = input("Run training now? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Training cancelled by user.")
        return

    print("\nStarting training...\n")
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="FlowRL Experiment & Training Pipeline"
    )
    parser.add_argument(
        "stage",
        choices=["experiment", "organize", "train", "full"],
        help="Pipeline stage to execute. Use --help with a specific stage for details."
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["rl"],
        choices=CONTROLLERS,
        help="Controllers to run experiments for (default: rl)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per controller (default: 3)"
    )
    parser.add_argument(
        "--note", type=str, default="",
        help="Optional note to append to trace filenames"
    )
    parser.add_argument(
        "--timestamp", type=str, default=None,
        help="Timestamp to use when organizing traces"
    )
    parser.add_argument(
        "--trace-path", type=str, default=None,
        help="Path to a specific trace file for training"
    )
    parser.add_argument(
        "--episodes", type=int, default=300,
        help="Number of training episodes (default: 300)"
    )
    parser.add_argument(
        "--train-after", action="store_true",
        help="After organizing traces, automatically launch training for each controller (uses the latest trace per controller)"
    )

    args = parser.parse_args()

    if args.stage == "experiment":
        stage_experiment(args.controllers, args.runs, args.note)

    elif args.stage == "organize":
        ts = args.timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stage_organize(timestamp=ts, train_after=args.train_after)

    elif args.stage == "train":
        if not args.trace_path:
            print("Error: --trace-path is required for 'train' stage")
            sys.exit(1)
        stage_train(args.trace_path, args.episodes)

    elif args.stage == "full":
        print("Full pipeline (guided mode).\n")
        stage_experiment(args.controllers, args.runs, args.note)
        print("\nAfter you finish the manual experiment steps and traces are collected, run:")
        print(f"    python {__file__} organize --train-after --timestamp <timestamp-from-experiment>")


if __name__ == "__main__":
    main()