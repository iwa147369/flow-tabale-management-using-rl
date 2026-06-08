import gymnasium as gym
import numpy as np
from tabulate import tabulate

# Optional: Trace-based realistic reward (Phase 0.5+)
try:
    from training.trace_simulator import TraceSimulator, generate_synthetic_trace_from_universe
except Exception:
    TraceSimulator = None
    generate_synthetic_trace_from_universe = None

TABLE_SIZE = 100
NUM_FLOWS  = 1000

# Flow universe — unique (src,dst) identities the simulation draws from
FLOW_UNIVERSE_SIZE = 300
HOT_FLOW_FRACTION  = 0.20   # 20% of unique flows are "hot"
HOT_FLOW_WEIGHT    = 0.80   # hot flows account for 80% of arrivals

MISS_WINDOW = 50  # steps within which a returning evicted flow counts as a miss


def build_flow_universe(size=FLOW_UNIVERSE_SIZE):
    """Create a stable pool of unique flow identities (simulated src/dst pairs)."""
    hot_count = int(size * HOT_FLOW_FRACTION)
    return [{'flow_id': i, 'is_hot': i < hot_count} for i in range(size)]


def generate_flows(num_flows, flow_universe):
    """Sample flows from the universe with Zipf-like weights.

    Hot flows (20% of universe) account for 80% of arrivals.
    Feature values vary per arrival — age/traffic change each time the same
    flow is seen, which is realistic. flow_id is stable across appearances.
    """
    n = len(flow_universe)
    hot_count  = sum(1 for f in flow_universe if f['is_hot'])
    cold_count = n - hot_count

    weights = np.array(
        [HOT_FLOW_WEIGHT / hot_count]          * hot_count +
        [(1 - HOT_FLOW_WEIGHT) / cold_count]   * cold_count,
        dtype=np.float64,
    )
    weights /= weights.sum()

    indices = np.random.choice(n, size=num_flows, p=weights)

    flows = []
    for idx in indices:
        entry = flow_universe[idx]
        # Hot flows skew toward higher priority; cold flows are mostly priority 1
        if entry['is_hot']:
            priority = int(np.random.choice([2, 5, 10], p=[0.50, 0.30, 0.20]))
        else:
            priority = int(np.random.choice([1, 2], p=[0.85, 0.15]))

        timeout     = int(min(np.random.exponential(60), 600))
        packet_count = int(np.clip(np.random.pareto(2.0) * 5 + 1, 1, 100_000))
        bytes_count  = packet_count * int(np.random.uniform(64, 1500))

        flows.append({
            'flow_id':     entry['flow_id'],
            'is_hot':      entry['is_hot'],
            'priority':    priority,
            'timeout':     timeout,
            'packet_count': packet_count,
            'bytes_count':  bytes_count,
        })
    return flows


class FlowTableEnvironment(gym.Env):
    def __init__(self, table_size=TABLE_SIZE, trace_simulator: "TraceSimulator" = None):
        super(FlowTableEnvironment, self).__init__()
        self.table_size = table_size
        # State: 4 features × table_size flows (identity not in state vector yet —
        # will be added when switching to pointer network architecture)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(table_size * 4,), dtype=np.float32
        )

        self.flow_universe = build_flow_universe()
        self.flow_table    = generate_flows(self.table_size, self.flow_universe)
        self.flows         = generate_flows(NUM_FLOWS, self.flow_universe)
        self.current_flow_index = 0
        self.step_count         = 0
        self.recently_evicted   = {}   # {flow_id: step_when_evicted}
        self.miss_count         = 0

        # --- Trace-based realistic reward (Phase 0.5) ---
        self.trace_simulator = trace_simulator
        self.use_trace_reward = trace_simulator is not None
        if self.use_trace_reward:
            print("[Env] Using TraceSimulator for delayed observed rewards (more realistic)")

    def reset(self):
        self.flow_table         = generate_flows(self.table_size, self.flow_universe)
        self.flows              = generate_flows(NUM_FLOWS, self.flow_universe)
        self.recently_evicted   = {}
        self.miss_count         = 0
        self.step_count         = 0
        self.current_flow_index = 0
        return self._get_state()

    def _get_state(self):
        """400-dim normalised state vector — 4 features per flow, order-dependent (known gap)."""
        max_priority = max(f['priority']     for f in self.flow_table) + 1e-6
        max_timeout  = max(f['timeout']      for f in self.flow_table) + 1e-6
        max_packets  = max(f['packet_count'] for f in self.flow_table) + 1e-6
        max_bytes    = max(f['bytes_count']  for f in self.flow_table) + 1e-6

        flow_info = []
        for f in self.flow_table:
            flow_info.append(f['priority']     / max_priority)
            flow_info.append(f['timeout']      / max_timeout)
            flow_info.append(f['packet_count'] / max_packets)
            flow_info.append(f['bytes_count']  / max_bytes)

        return np.array(flow_info, dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        # Select which flow to evict
        if action == 0:
            flow_index = _argmin(self.flow_table, 'priority')
        elif action == 1:
            flow_index = _argmax(self.flow_table, 'timeout')
        elif action == 2:
            flow_index = _argmin(self.flow_table, 'packet_count')
        else:  # action == 3
            flow_index = _argmin(self.flow_table, 'bytes_count')

        incoming = self.flows[self.current_flow_index]

        # Miss detection: incoming flow was recently evicted
        miss = incoming['flow_id'] in self.recently_evicted
        if miss:
            self.miss_count += 1
            del self.recently_evicted[incoming['flow_id']]

        reward = self._calculate_reward(flow_index, miss)

        # Record evicted flow's identity for future miss detection
        self.recently_evicted[self.flow_table[flow_index]['flow_id']] = self.step_count

        # Prune stale entries outside the miss window
        cutoff = self.step_count - MISS_WINDOW
        self.recently_evicted = {
            fid: s for fid, s in self.recently_evicted.items() if s > cutoff
        }

        self.flow_table.pop(flow_index)
        self.flow_table.append(incoming)
        self.current_flow_index += 1

        done = self.current_flow_index >= len(self.flows)
        info = {
            "miss_count": self.miss_count,
            "total_steps": self.step_count,
        }
        return self._get_state(), reward, done, info

    def _calculate_reward(self, flow_index, miss=False):
        f = self.flow_table[flow_index]
        n = len(self.flow_table)

        avg_priority = sum(x['priority']     for x in self.flow_table) / n
        avg_timeout  = sum(x['timeout']      for x in self.flow_table) / n
        avg_packets  = sum(x['packet_count'] for x in self.flow_table) / n
        avg_bytes    = sum(x['bytes_count']  for x in self.flow_table) / n

        def zone(val, avg):
            if val > avg * 1.25: return -1
            if val < avg * 0.5:  return  1
            return 0

        heuristic = (
            zone(f['priority'],     avg_priority) * 0.3 +
            zone(f['timeout'],      avg_timeout)  * 0.3 +
            zone(f['packet_count'], avg_packets)  * 0.2 +
            zone(f['bytes_count'],  avg_bytes)    * 0.2
        )
        miss_penalty = -1.0 if miss else 0.0

        # --- Trace-based realistic penalty (Phase 0.5) ---
        trace_penalty = 0.0
        if self.use_trace_reward and self.trace_simulator is not None:
            # Ask the simulator: "What was the real future cost of evicting this flow now?"
            penalty, _ = self.trace_simulator.compute_miss_penalty(
                f['flow_id'], self.step_count, window=50
            )
            trace_penalty = penalty

        return heuristic + miss_penalty + trace_penalty

    def print_flow_table(self):
        rows = []
        for i, f in enumerate(self.flow_table):
            rows.append({'index': i, 'flow_id': f['flow_id'], 'hot': f['is_hot'],
                         'priority': f['priority'], 'timeout': f['timeout'],
                         'packets': f['packet_count'], 'bytes': f['bytes_count']})
        print(tabulate(rows, headers='keys', tablefmt='grid'))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _argmin(flow_table, key):
    """Return index of flow with minimum value for key."""
    return min(range(len(flow_table)), key=lambda i: flow_table[i][key])


def _argmax(flow_table, key):
    """Return index of flow with maximum value for key."""
    return max(range(len(flow_table)), key=lambda i: flow_table[i][key])
