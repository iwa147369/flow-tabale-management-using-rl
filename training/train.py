import os

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from training.environment import FlowTableEnvironment
from training.model import DoubleDQNAgent, ReplayBuffer, BATCH_SIZE

# For trace-based reward experiments
try:
    from training.trace_simulator import TraceSimulator, generate_synthetic_trace_from_universe
except Exception:
    TraceSimulator = None
    generate_synthetic_trace_from_universe = None


def train_agent(env, agent, episodes=1000, batch_size=BATCH_SIZE,
                epsilon_decay=0.995, epsilon_min=0.01, gamma=None):
    """
    Main training loop.

    New configurable parameters (Phase 0, June 2026):
    - epsilon_decay / epsilon_min : allow slower exploration for harder objectives
    - gamma : override the agent's discount factor at runtime
    """
    if gamma is not None:
        agent.gamma = gamma

    if _WANDB:
        wandb.init(project="flow-table-management-v3",
                   config={
                       "episodes": episodes,
                       "batch_size": batch_size,
                       "epsilon_decay": epsilon_decay,
                       "epsilon_min": epsilon_min,
                       "gamma": agent.gamma,
                       "table_size": env.table_size,
                   })

    epsilon = 1.0
    replay_buffer = ReplayBuffer()
    episode_rewards = []
    loss_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_loss = 0
        steps = 0

        episode_misses = 0
        while not done:
            action = agent.act(state, epsilon)
            result = env.step(action)
            if len(result) == 5:  # Modern Gymnasium (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
                episode_misses = info.get("miss_count", 0)
            else:  # Compatibility with older 3 or 4-tuple style
                if len(result) == 4:
                    next_state, reward, done, info = result
                    episode_misses = info.get("miss_count", 0)
                else:
                    next_state, reward, done = result
            total_reward += reward
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            if len(replay_buffer) >= batch_size:
                loss = agent.learn(replay_buffer, batch_size)
                if loss is not None:
                    episode_loss += loss

        avg_loss = episode_loss / steps if steps > 0 else 0
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        episode_rewards.append(total_reward)
        loss_history.append(avg_loss)

        if _WANDB:
            wandb.log({
                "total_reward": total_reward,
                "episode_misses": episode_misses,
                "average_loss": avg_loss,
                "epsilon": epsilon
            })

        print(f"Episode {episode + 1:4d} | reward {total_reward:7.2f} | "
              f"misses {episode_misses:3d} | loss {avg_loss:.4f} | epsilon {epsilon:.4f}")

        if (episode + 1) % 100 == 0:
            save_path = f'models/model_episode_{episode + 1}.pt'
            agent.save_model(save_path)
            print(f"  --> Model saved to {save_path}")

    if _WANDB:
        wandb.finish()
    return agent, episode_rewards, loss_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Flow Table RL agent")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override discount factor (default from agent)")
    parser.add_argument("--use-trace-reward", action="store_true",
                        help="Use TraceSimulator for more realistic delayed rewards (Phase 0.5 experiment)")
    args = parser.parse_args()

    trace_sim = None
    if args.use_trace_reward:
        if generate_synthetic_trace_from_universe is None:
            print("Warning: trace_simulator module not available. Falling back to heuristic reward.")
        else:
            print("Generating improved synthetic trace for realistic delayed rewards...")
            temp_env = FlowTableEnvironment()
            synthetic_traces = generate_synthetic_trace_from_universe(
                temp_env.flow_universe,
                num_steps=8000,
                seed=42
            )
            trace_sim = TraceSimulator(synthetic_traces)

            n_flows = len(synthetic_traces)
            total_arrivals = sum(len(t.arrivals) for t in synthetic_traces.values())
            print(f"Synthetic trace ready: {n_flows} unique flows, {total_arrivals} total arrivals")
            print("Using TraceSimulator-based rewards (more honest future-cost signal).\n")

    env = FlowTableEnvironment(trace_simulator=trace_sim)
    action_size = env.action_space.n
    agent = DoubleDQNAgent(env.observation_space.shape[0], action_size)

    train_agent(
        env, agent,
        episodes=args.episodes,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        gamma=args.gamma
    )
