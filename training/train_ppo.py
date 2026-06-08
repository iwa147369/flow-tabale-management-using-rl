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

from training.set_environment import SetFlowTableEnvironment, NUM_FEATURES
from training.ppo_model import PPOAgent

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
        misses = info.get("miss_count", misses)
        obs = next_obs

    # bootstrap value for the final (terminal) state is 0
    return roll, total_reward, misses


def build_trace_sim(args, env):
    if not args.use_trace_reward and not args.trace_file:
        return None
    if TraceSimulator is None:
        print("Warning: trace_simulator unavailable; falling back to heuristic reward.")
        return None

    if args.trace_file:
        print(f"Loading real trace from {args.trace_file}")
        rec = TraceRecorder.load(args.trace_file)
        traces = rec.get_trace()
    else:
        print("Generating improved synthetic trace for realistic delayed rewards...")
        traces = generate_synthetic_trace_from_universe(env.flow_universe, num_steps=8000, seed=42)

    total_arrivals = sum(len(t.arrivals) for t in traces.values())
    print(f"Trace ready: {len(traces)} unique flows, {total_arrivals} total arrivals\n")
    return TraceSimulator(traces)


def main():
    p = argparse.ArgumentParser(description="Train pointer-network PPO flow-table agent")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--use-trace-reward", action="store_true",
                   help="Use synthetic TraceSimulator delayed rewards")
    p.add_argument("--trace-file", type=str, default=None,
                   help="Path to a real recorded trace .pkl (overrides synthetic)")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--save-prefix", type=str, default="models/ppo_model")
    args = p.parse_args()

    # A bare env first so we can build the trace simulator from its flow universe,
    # then attach the simulator to the real training env.
    base_env = SetFlowTableEnvironment()
    trace_sim = build_trace_sim(args, base_env)
    env = SetFlowTableEnvironment(trace_simulator=trace_sim)

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
