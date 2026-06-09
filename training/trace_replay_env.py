"""
Trace-replay environment for honest real-trace training.

Background: the synthetic SetFlowTableEnvironment generates its own flow stream
(ids 0..299, step-indexed time) and only *looks up* a reward in the trace. Recorded
traces are keyed by hashed match ids (flow_id_of) with second-timestamps, so that
lookup never matched — the delayed penalty was always 0 (a no-op). See memory.md
Work Log 2026-06-09.

This env fixes that by stepping through the recorded *arrival sequence itself*, using
the trace's own flow ids. The eviction cost is a **reuse-distance** signal measured in
the event stream (how many upcoming arrivals until the evicted flow is requested
again), traffic-weighted. We use event-stream distance rather than wall-clock seconds
because the recorded arrivals are bursty (consecutive re-arrivals of a flow are ~0s
apart), so a seconds-based window carries no signal.

Semantics:
- One gym step = one eviction *decision*: a miss into a full table. Hits and inserts
  into a not-yet-full table are fast-forwarded internally.
- Episode = up to `max_decisions` decisions from a random offset (sliding start), ends
  early if the trace is exhausted.
- Reward = penalty - evict_cost, where penalty < 0 iff the evicted flow is requested
  again within `reuse_window` upcoming arrivals; magnitude = normalized avg traffic of
  the flow, scaled so sooner reuse hurts more. No heuristic shaping.

Caveat (documented, not fixed here): the trace is an install/miss stream recorded
under FIFO/LRU/RL, not a pure per-packet demand stream, so the replayed workload is
mildly policy-dependent. A clean fix would record every PacketIn and re-collect.
"""

import gymnasium as gym
import numpy as np

NUM_FEATURES = 5          # identity, age, recency, avg_bytes, avg_packets
DEFAULT_TABLE_SIZE = 100
DEFAULT_MAX_DECISIONS = 1000
DEFAULT_REUSE_WINDOW = 200   # upcoming arrivals within which a return counts as a "miss"
DEFAULT_EVICT_COST = 0.02


class TraceReplayEnvironment(gym.Env):
    def __init__(self, trace, table_size=DEFAULT_TABLE_SIZE,
                 max_decisions=DEFAULT_MAX_DECISIONS, reuse_window=DEFAULT_REUSE_WINDOW,
                 evict_cost=DEFAULT_EVICT_COST, seed=None):
        super().__init__()
        self.table_size = table_size
        self.max_decisions = max_decisions
        self.reuse_window = reuse_window
        self.evict_cost = evict_cost
        self.rng = np.random.default_rng(seed)

        # Flatten the trace into a single time-ordered arrival stream.
        events = [(t, fid) for fid, ft in trace.items() for t in ft.arrivals]
        events.sort(key=lambda e: e[0])
        self.events = events
        self.event_fids = [fid for _, fid in events]

        # next_same[i] = index of the next arrival of the SAME flow after i (else len).
        # Lets us measure reuse distance in O(1) at eviction time.
        n = len(events)
        self.next_same = [n] * n
        last = {}
        for i in range(n - 1, -1, -1):
            fid = self.event_fids[i]
            self.next_same[i] = last.get(fid, n)
            last[fid] = i

        # Per-flow static features (lifetime averages) + a stable normalized identity.
        ids = sorted(trace.keys())
        self.id_index = {fid: i for i, fid in enumerate(ids)}
        self.num_unique = max(1, len(ids))
        self.avg_bytes, self.avg_packets = {}, {}
        for fid, ft in trace.items():
            k = max(1, len(ft.arrivals))
            self.avg_bytes[fid] = ft.total_bytes / k
            self.avg_packets[fid] = ft.total_packets / k
        self._max_avg_bytes = (max(self.avg_bytes.values()) + 1e-6) if self.avg_bytes else 1.0
        self._max_avg_packets = (max(self.avg_packets.values()) + 1e-6) if self.avg_packets else 1.0

        # Rank-based traffic weight in [0,1] for the reward — robust to the heavy-tailed
        # byte distribution (max-normalization crushed almost every flow to ~0, leaving
        # no usable reward signal). Rank gives a well-spread weight across flows.
        order = sorted(self.avg_bytes, key=self.avg_bytes.get)
        self.bytes_rank = {fid: (i / max(1, len(order) - 1)) for i, fid in enumerate(order)}

        self.action_space = gym.spaces.Discrete(table_size)
        self.observation_space = gym.spaces.Dict({
            "table":    gym.spaces.Box(0, 1, shape=(table_size, NUM_FEATURES), dtype=np.float32),
            "incoming": gym.spaces.Box(0, 1, shape=(NUM_FEATURES,), dtype=np.float32),
            "mask":     gym.spaces.Box(0, 1, shape=(table_size,), dtype=np.float32),
        })

    # ── internals ───────────────────────────────────────────────────────────
    def _install(self, fid, t, idx):
        self.table[fid] = {"install_time": t, "last_seen": t, "last_idx": idx}

    def _advance_to_decision(self):
        """Fast-forward through hits / not-full inserts until a miss into a full
        table (a decision) or the stream ends. Returns True if a decision is pending."""
        n = len(self.events)
        while self.ptr < n:
            t, fid = self.events[self.ptr]
            if fid in self.table:                       # hit
                self.table[fid]["last_seen"] = t
                self.table[fid]["last_idx"] = self.ptr
                self.ptr += 1
                continue
            if len(self.table) < self.table_size:       # miss, room available
                self._install(fid, t, self.ptr)
                self.ptr += 1
                continue
            self.current_time = t                        # miss + full → decision
            self.incoming_fid = fid
            return True
        return False

    def _feat(self, fid, age, recency):
        return [
            self.id_index.get(fid, 0) / self.num_unique,
            age,
            recency,
            self.avg_bytes.get(fid, 0.0),
            self.avg_packets.get(fid, 0.0),
        ]

    def _zero_obs(self):
        return {
            "table": np.zeros((self.table_size, NUM_FEATURES), dtype=np.float32),
            "incoming": np.zeros(NUM_FEATURES, dtype=np.float32),
            "mask": np.ones(self.table_size, dtype=np.float32),
        }

    def _get_obs(self):
        resident = list(self.table.keys())   # exactly table_size at a decision point
        t = self.current_time
        raw = np.array(
            [self._feat(fid, t - m["install_time"], t - m["last_seen"])
             for fid, m in self.table.items()],
            dtype=np.float64,
        )
        table = np.zeros((self.table_size, NUM_FEATURES), dtype=np.float32)
        if len(raw):
            for col in (1, 2):               # age, recency: normalize by max-in-table
                raw[:, col] = raw[:, col] / (raw[:, col].max() + 1e-6)
            raw[:, 3] = raw[:, 3] / self._max_avg_bytes
            raw[:, 4] = raw[:, 4] / self._max_avg_packets
            table[:len(raw)] = raw.astype(np.float32)

        inc = np.array(self._feat(self.incoming_fid, 0.0, 0.0), dtype=np.float64)
        inc[3] /= self._max_avg_bytes
        inc[4] /= self._max_avg_packets

        self._resident = resident
        return {
            "table": table,
            "incoming": inc.astype(np.float32),
            "mask": np.ones(self.table_size, dtype=np.float32),  # trace has no critical flows
        }

    def _eviction_penalty(self, evicted_fid):
        """Reuse-distance cost: negative iff the evicted flow is requested again within
        `reuse_window` upcoming arrivals; 0 for a clean eviction. Magnitude scales with
        how soon it returns and its traffic rank — with a floor so that reusing *any*
        flow soon still costs something, and high-traffic flows cost the most."""
        nxt = self.next_same[self.table[evicted_fid]["last_idx"]]
        reuse = nxt - self.ptr                      # arrivals until it is needed again
        if reuse > self.reuse_window:
            return 0.0
        recency_factor = 1.0 - reuse / self.reuse_window      # in (0, 1], sooner = bigger
        traffic_weight = 0.25 + 0.75 * self.bytes_rank.get(evicted_fid, 0.0)  # [0.25, 1.0]
        return -recency_factor * traffic_weight

    # ── gym API ───────────────────────────────────────────────────────────
    def reset(self):
        self.table = {}
        self.decisions = 0
        if len(self.events) > 10:            # sliding random start for variety
            self.ptr = int(self.rng.integers(0, max(1, len(self.events) // 2)))
        else:
            self.ptr = 0
        if not self._advance_to_decision():  # random offset overshot → restart clean
            self.table = {}
            self.ptr = 0
            self._advance_to_decision()
        return self._get_obs()

    def step(self, action):
        resident = self._resident
        idx = int(action)
        if idx < 0 or idx >= len(resident):
            idx = 0
        evicted = resident[idx]

        reward = self._eviction_penalty(evicted) - self.evict_cost

        del self.table[evicted]
        self._install(self.incoming_fid, self.current_time, self.ptr)
        self.ptr += 1
        self.decisions += 1

        more = self._advance_to_decision()
        done = (self.decisions >= self.max_decisions) or (not more)
        obs = self._get_obs() if more else self._zero_obs()
        info = {"decisions": self.decisions, "evicted": evicted}
        return obs, reward, done, info
