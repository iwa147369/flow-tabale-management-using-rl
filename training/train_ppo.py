"""
PPO training loop for the pointer-network flow-table agent (Phase 3).

Runnable today against the synthetic TraceSimulator (no Mininet needed):

    .venv/bin/python -m training.train_ppo --episodes 300 --use-trace-reward

Once you have a real Mininet trace (traces/collected/...), train on it with:

    .venv/bin/python -m training.train_ppo --episodes 1000 \
        --trace-file traces/collected/rl/trace_rl_<date>.pkl

The architecture and reward are identical between synthetic and real runs — only
the trace source changes, so a synthetic-trained policy can be re-run on real
traces for the honest comparison.
"""

import argparse
import datetime
import os
import sys

from training.set_environment import SetFlowTableEnvironment, NUM_FEATURES
from training.trace_replay_env import TraceReplayEnvironment
from training.ppo_model import PPOAgent

LOG_DIR = "logs"


class _Tee:
    """Write print() output to the terminal and a log file at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def _start_run_log(args):
    """Open a uniquely-named log for this run under logs/ and tee stdout to it.

    One file per run (named by trace source + timestamp), so logs never clobber
    each other and the repo root stays clean. Returns the open file handle."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if args.trace_file:
        source = os.path.splitext(os.path.basename(args.trace_file))[0]
    elif args.use_trace_reward:
        source = "synthetic_trace"
    else:
        source = "heuristic"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"ppo_{source}_{stamp}.log")
    f = open(path, "w")
    sys.stdout = _Tee(sys.__stdout__, f)
    print(f"[log] this run is being saved to {path}")
    return f

try:
    from training.trace_simulator import (
        TraceRecorder, TraceSimulator, generate_synthetic_trace_from_universe,
    )
except Exception:
    TraceRecorder = TraceSimulator = generate_synthetic_trace_from_universe = None


def collect_episode(env, agent):
    """One full episode rollout."""
    obs = env.reset()
    roll = {"obs": [], "actions": [], "logps": [], "rewards": [], "values": [], "dones": []}
    total_reward = 0.0
    misses = 0
    done = False
    while not done:
        action, logp, value = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        roll["obs"].append(obs)
        roll["actions"].append(action)
        roll["logps"].append(logp)
        roll["rewards"].append(reward)
        roll["values"].append(value)
        roll["dones"].append(1.0 if done else 0.0)
        total_reward += reward
        # synthetic env reports "miss_count"; trace-replay reports "decisions" (= ep length)
        misses = info.get("miss_count", info.get("decisions", misses))
        obs = next_obs

    # bootstrap value for the final (terminal) state is 0
    return roll, total_reward, misses


def build_env(args):
    """Pick the training environment.

    - --trace-file  → TraceReplayEnvironment: replays the recorded arrivals using the
      trace's own flow ids and second-timestamps, so the delayed penalty actually fires.
    - --use-trace-reward → synthetic SetFlowTableEnvironment with delayed reward (ids 0..299,
      step-time — this lookup path only works for the synthetic generator).
    - neither → synthetic SetFlowTableEnvironment, heuristic + miss reward only.
    """
    if args.trace_file:
        if TraceRecorder is None:
            raise SystemExit("trace_simulator module unavailable; cannot load --trace-file")
        print(f"Loading real trace from {args.trace_file}")
        trace = TraceRecorder.load(args.trace_file).get_trace()
        total = sum(len(t.arrivals) for t in trace.values())
        print(f"Trace ready: {len(trace)} unique flows, {total} total arrivals")
        print("Using TraceReplayEnvironment (real arrivals; ids/time consistent).\n")
        return TraceReplayEnvironment(trace, table_size=args.table_size,
                                      max_decisions=args.max_decisions, reuse_window=args.reuse_window)

    if args.use_trace_reward and generate_synthetic_trace_from_universe is not None:
        print("Generating improved synthetic trace for realistic delayed rewards...")
        base = SetFlowTableEnvironment(table_size=args.table_size)
        traces = generate_synthetic_trace_from_universe(base.flow_universe, num_steps=8000, seed=42)
        total = sum(len(t.arrivals) for t in traces.values())
        print(f"Trace ready: {len(traces)} unique flows, {total} total arrivals\n")
        return SetFlowTableEnvironment(table_size=args.table_size, trace_simulator=TraceSimulator(traces))

    return SetFlowTableEnvironment(table_size=args.table_size)


def main():
    p = argparse.ArgumentParser(description="Train pointer-network PPO flow-table agent")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=None,
                   help="Discount. Default: 0.9 for --trace-file (near-immediate reward), else 0.99")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate. Default: 1e-3 for --trace-file, else 3e-4")
    p.add_argument("--use-trace-reward", action="store_true",
                   help="Use synthetic TraceSimulator delayed rewards")
    p.add_argument("--trace-file", type=str, default=None,
                   help="Path to a real recorded trace .pkl → trains via TraceReplayEnvironment")
    p.add_argument("--table-size", type=int, default=100,
                   help="Flow-table capacity (set to the real switch limit)")
    p.add_argument("--reuse-window", type=int, default=200,
                   help="Reuse window in # of upcoming arrivals (trace-replay reward)")
    p.add_argument("--max-decisions", type=int, default=1000,
                   help="Eviction decisions per episode (trace-replay)")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--save-prefix", type=str, default="models/ppo_model")
    args = p.parse_args()

    _start_run_log(args)

    # Trace-replay reward is near-immediate (a per-step bandit), so it wants a lower
    # discount + higher lr than the synthetic long-horizon reward.
    if args.gamma is None:
        args.gamma = 0.9 if args.trace_file else 0.99
    if args.lr is None:
        args.lr = 1e-3 if args.trace_file else 3e-4
    print(f"[config] gamma={args.gamma} lr={args.lr}")

    env = build_env(args)
    agent = PPOAgent(num_features=NUM_FEATURES, gamma=args.gamma, lr=args.lr)

    recent = []
    for ep in range(args.episodes):
        roll, total_reward, misses = collect_episode(env, agent)
        stats = agent.update(roll, last_value=0.0)
        recent.append(total_reward)
        recent = recent[-50:]

        print(f"Episode {ep + 1:4d} | reward {total_reward:8.2f} | "
              f"misses {misses:3d} | "
              f"ploss {stats.get('policy_loss', 0):+.4f} | "
              f"vloss {stats.get('value_loss', 0):.3f} | "
              f"entropy {stats.get('entropy', 0):.3f} | "
              f"avg50 {sum(recent)/len(recent):8.2f}")

        if (ep + 1) % args.save_every == 0:
            path = f"{args.save_prefix}_episode_{ep + 1}.pt"
            agent.save_model(path)
            print(f"  --> Model saved to {path}")


if __name__ == "__main__":
    main()
