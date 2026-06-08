"""
Set-structured flow-table environment for the pointer-network / PPO architecture
(Phase 3 of the redesign in docs/memory.md).

Differences from the v1 FlowTableEnvironment (training/environment.py):
- Observation is *set-structured*, not a flat 400-dim vector:
      table:    (table_size, NUM_FEATURES)   permutation-invariant set of flows
      incoming: (NUM_FEATURES,)              the flow that triggered the eviction
      mask:     (table_size,)                1 = evictable, 0 = protected/critical
- Action is the *index of the flow to evict* (Discrete(table_size)) — direct
  selection, not a choice among 4 fixed heuristics. This is what the professor
  originally asked for and what the pointer network is built to express.
- Identity (flow_id) is now part of the per-flow feature vector, so the policy
  can in principle learn "protect this specific hot flow".

Reward reuses the same honest signal as v1: heuristic zone score + miss penalty
+ optional trace-based delayed penalty (TraceSimulator). When a real Mininet
trace is supplied, the delayed penalty dominates.
"""

import gymnasium as gym
import numpy as np

from training.environment import (
    build_flow_universe,
    generate_flows,
    TABLE_SIZE,
    NUM_FLOWS,
    MISS_WINDOW,
    FLOW_UNIVERSE_SIZE,
)

try:
    from training.trace_simulator import TraceSimulator
except Exception:
    TraceSimulator = None

# priority, age(timeout), packet_count, bytes_count, identity
NUM_FEATURES = 5


class SetFlowTableEnvironment(gym.Env):
    """Pointer-network friendly flow-table env. Action = index of flow to evict."""

    def __init__(self, table_size=TABLE_SIZE, trace_simulator: "TraceSimulator" = None):
        super().__init__()
        self.table_size = table_size
        self.num_features = NUM_FEATURES

        # Action: which of the `table_size` resident flows to evict.
        self.action_space = gym.spaces.Discrete(table_size)
        self.observation_space = gym.spaces.Dict({
            "table":    gym.spaces.Box(0, 1, shape=(table_size, NUM_FEATURES), dtype=np.float32),
            "incoming": gym.spaces.Box(0, 1, shape=(NUM_FEATURES,), dtype=np.float32),
            "mask":     gym.spaces.Box(0, 1, shape=(table_size,), dtype=np.float32),
        })

        self.flow_universe = build_flow_universe()
        self.trace_simulator = trace_simulator
        self.use_trace_reward = trace_simulator is not None
        if self.use_trace_reward:
            print("[SetEnv] Using TraceSimulator for delayed observed rewards")

        self.reset()

    # ── Gym API ────────────────────────────────────────────────────────────
    def reset(self):
        self.flow_table         = generate_flows(self.table_size, self.flow_universe)
        self.flows              = generate_flows(NUM_FLOWS, self.flow_universe)
        self.recently_evicted   = {}
        self.miss_count         = 0
        self.step_count         = 0
        self.current_flow_index = 0
        return self._get_obs()

    def step(self, action):
        self.step_count += 1
        flow_index = int(action)
        # Defensive: an invalid / masked index falls back to the safest heuristic
        # (evict the lowest-priority flow) so a stray action can never crash a rollout.
        if flow_index < 0 or flow_index >= len(self.flow_table) or not self._evictable(flow_index):
            flow_index = self._fallback_index()

        incoming = self.flows[self.current_flow_index]

        miss = incoming['flow_id'] in self.recently_evicted
        if miss:
            self.miss_count += 1
            del self.recently_evicted[incoming['flow_id']]

        reward = self._calculate_reward(flow_index, miss)

        self.recently_evicted[self.flow_table[flow_index]['flow_id']] = self.step_count
        cutoff = self.step_count - MISS_WINDOW
        self.recently_evicted = {
            fid: s for fid, s in self.recently_evicted.items() if s > cutoff
        }

        self.flow_table.pop(flow_index)
        self.flow_table.append(incoming)
        self.current_flow_index += 1

        done = self.current_flow_index >= len(self.flows)
        info = {"miss_count": self.miss_count, "total_steps": self.step_count}
        return self._get_obs(), reward, done, info

    # ── Critical-flow protection (GAP 4 mechanism) ──────────────────────────
    def _evictable(self, idx):
        """Hard pre-filter hook. Priority 0 (table-miss) and 65535 are never
        eviction candidates. Synthetic flows never hit these, but the mechanism
        is wired so real traces / controller flows get protected for free."""
        p = self.flow_table[idx]['priority']
        return p != 0 and p != 65535

    def _eviction_mask(self):
        return np.array(
            [1.0 if self._evictable(i) else 0.0 for i in range(len(self.flow_table))],
            dtype=np.float32,
        )

    def _fallback_index(self):
        evictable = [i for i in range(len(self.flow_table)) if self._evictable(i)]
        if not evictable:
            return 0
        return min(evictable, key=lambda i: self.flow_table[i]['priority'])

    # ── Observation ─────────────────────────────────────────────────────────
    def _feature_row(self, f, maxes):
        mp, mt, mpk, mb = maxes
        return [
            f['priority']     / mp,
            f['timeout']      / mt,
            f['packet_count'] / mpk,
            f['bytes_count']  / mb,
            f['flow_id']      / FLOW_UNIVERSE_SIZE,
        ]

    def _get_obs(self):
        maxes = (
            max(f['priority']     for f in self.flow_table) + 1e-6,
            max(f['timeout']      for f in self.flow_table) + 1e-6,
            max(f['packet_count'] for f in self.flow_table) + 1e-6,
            max(f['bytes_count']  for f in self.flow_table) + 1e-6,
        )
        table = np.array([self._feature_row(f, maxes) for f in self.flow_table],
                         dtype=np.float32)

        incoming_flow = self.flows[self.current_flow_index] if self.current_flow_index < len(self.flows) else self.flow_table[0]
        incoming = np.array(self._feature_row(incoming_flow, maxes), dtype=np.float32)

        return {"table": table, "incoming": incoming, "mask": self._eviction_mask()}

    # ── Reward (same honest signal as v1, centred on the chosen flow) ───────
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

        trace_penalty = 0.0
        if self.use_trace_reward and self.trace_simulator is not None:
            penalty, _ = self.trace_simulator.compute_miss_penalty(
                f['flow_id'], self.step_count, window=50
            )
            trace_penalty = penalty

        return heuristic + miss_penalty + trace_penalty
