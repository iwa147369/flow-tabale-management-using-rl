"""
Trace Recorder + Simulator for realistic delayed rewards (Phase 2 foundation).

The core idea:
- Record real traffic traces from Mininet/Ryu (flow arrivals, sizes, durations).
- Replay them at high speed in pure Python.
- When the agent evicts a flow, the simulator can tell us:
    * Did this flow return later?
    * How much traffic did it carry after returning?
  → This becomes the true observed reward instead of our current synthetic heuristic.

This module is the scaffolding. Real recording hooks (in Ryu or ovs-ofctl) will be added later.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class FlowEvent:
    """A single observation of a flow at a point in time."""
    flow_id: int
    timestamp: float          # seconds since trace start
    bytes: int = 0
    packets: int = 0
    duration: float = 0.0     # how long the flow has been alive at this observation


@dataclass
class FlowTrace:
    """Complete lifetime information for one flow in the trace."""
    flow_id: int
    arrivals: List[float] = field(default_factory=list)   # every time we saw the flow arrive/install
    total_bytes: int = 0
    total_packets: int = 0
    first_seen: float = float("inf")
    last_seen: float = 0.0


class TraceRecorder:
    """
    Records live traffic into a structured trace.

    Later we will hook this from the Ryu controller or from a Mininet experiment script.
    For now it is a manual API that can be driven from Python.
    """

    def __init__(self):
        self.start_time = time.time()
        self.flows: Dict[int, FlowTrace] = {}
        self.raw_events: List[FlowEvent] = []

    def record_flow(self, flow_id: int, bytes: int = 0, packets: int = 0, timestamp: Optional[float] = None):
        """Call this every time we observe a flow (on install, on stats reply, etc.)."""
        if timestamp is None:
            timestamp = time.time() - self.start_time

        if flow_id not in self.flows:
            self.flows[flow_id] = FlowTrace(flow_id=flow_id, first_seen=timestamp)

        ft = self.flows[flow_id]
        ft.arrivals.append(timestamp)
        ft.total_bytes += bytes
        ft.total_packets += packets
        ft.last_seen = max(ft.last_seen, timestamp)

        self.raw_events.append(FlowEvent(flow_id, timestamp, bytes, packets))

    def add_stats(self, flow_id: int, bytes: int = 0, packets: int = 0, timestamp: Optional[float] = None):
        """Add traffic totals for a flow WITHOUT recording a new arrival.

        Call this from byte-accounting events (e.g. an OpenFlow FlowRemoved, which
        carries the flow's lifetime byte/packet counts). Unlike record_flow, this
        does not append to `arrivals` — arrivals must stay equal to the number of
        times the flow (re)appeared, since the simulator uses them for return
        detection."""
        if timestamp is None:
            timestamp = time.time() - self.start_time

        if flow_id not in self.flows:
            self.flows[flow_id] = FlowTrace(flow_id=flow_id, first_seen=timestamp)

        ft = self.flows[flow_id]
        ft.total_bytes += bytes
        ft.total_packets += packets
        ft.last_seen = max(ft.last_seen, timestamp)

    def get_trace(self) -> Dict[int, FlowTrace]:
        return self.flows

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"flows": self.flows, "raw_events": self.raw_events}, f)
        print(f"Trace saved to {path} ({len(self.flows)} unique flows)")

    @staticmethod
    def load(path: str) -> "TraceRecorder":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        rec = TraceRecorder()
        rec.flows = data["flows"]
        rec.raw_events = data.get("raw_events", [])
        return rec


class TraceSimulator:
    """
    Fast offline simulator that replays a recorded trace and can answer
    "what would have happened if we evicted flow X at time T?" questions.

    This is where we will compute the real delayed reward signal.
    """

    def __init__(self, trace: Dict[int, FlowTrace]):
        self.trace = trace
        self.sorted_flows = sorted(trace.values(), key=lambda f: f.first_seen)

    def compute_miss_penalty(self, evicted_flow_id: int, eviction_time: float,
                             window: float = 50.0) -> Tuple[float, int]:
        """
        Given that we evicted `evicted_flow_id` at `eviction_time`,
        compute a realistic penalty based on future arrivals in the trace.

        This version works well with the improved synthetic trace generator.
        """
        if evicted_flow_id not in self.trace:
            return 0.0, 0

        ft = self.trace[evicted_flow_id]

        # Find arrivals that happened after the eviction
        future_arrivals = [t for t in ft.arrivals if t > eviction_time]
        if not future_arrivals:
            return 0.0, 0  # Clean eviction, flow never returned

        first_return = future_arrivals[0]
        if first_return - eviction_time > window:
            return 0.0, 0

        # Estimate traffic carried after return (within the window)
        # We approximate by taking the average bytes per arrival for this flow
        if len(ft.arrivals) > 0:
            avg_bytes_per_appearance = ft.total_bytes / max(1, len(ft.arrivals))
        else:
            avg_bytes_per_appearance = 5000

        # Rough count of how many times it returned inside the window
        returns_in_window = [t for t in future_arrivals if t <= eviction_time + window]
        bytes_after = int(len(returns_in_window) * avg_bytes_per_appearance)

        # Penalty: negative, scaled by traffic damage (tunable)
        penalty = - (bytes_after / 2_000_000.0)   # -0.5 per MB roughly

        return penalty, bytes_after

    def simulate_episode(self, evictions: List[Tuple[int, float]], window: float = 30.0) -> float:
        """
        Given a list of (flow_id, eviction_time) decisions, compute the total
        observed reward according to the trace.
        """
        total = 0.0
        for fid, t in evictions:
            pen, _ = self.compute_miss_penalty(fid, t, window=window)
            total += pen
        return total


# ------------------------------------------------------------------
# Synthetic Trace Generation (for bootstrapping experiments before real traces exist)
# ------------------------------------------------------------------

def generate_synthetic_trace_from_universe(
    flow_universe: list,
    num_steps: int = 8000,
    hot_interarrival_mean: float = 5.0,       # hot flows re-appear much more frequently (stronger bias)
    cold_interarrival_mean: float = 200.0,    # cold flows are very rare
    hot_lifetime_mean: float = 250.0,         # hot flows tend to live longer once active
    cold_lifetime_mean: float = 30.0,         # cold flows die out quickly
    bytes_per_second_pareto_alpha: float = 1.6,  # heavier tail for more extreme elephants
    seed: int = 42,
) -> Dict[int, FlowTrace]:
    """
    Generate a significantly more realistic synthetic trace from the hot/cold flow universe.

    Improvements over the basic version:
    - Models flows as having lifetimes (a flow is active for a period, then goes quiet).
    - Different inter-arrival and lifetime statistics for hot vs cold flows.
    - Heavy-tailed traffic volume (realistic elephant/mouse behavior).
    - Multiple arrivals per flow with temporal structure.
    - Produces traces where "return after eviction" behavior is meaningful for the simulator.

    This is currently the best way to experiment with honest delayed rewards
    without having real Mininet traces.
    """
    import numpy as np
    np.random.seed(seed)

    traces: Dict[int, FlowTrace] = {}

    # Precompute which flows are hot
    hot_ids = {f['flow_id'] for f in flow_universe if f.get('is_hot')}
    all_ids = [f['flow_id'] for f in flow_universe]

    # For each flow, decide its "sessions" across the trace horizon
    for fid in all_ids:
        is_hot = fid in hot_ids
        interarrival = hot_interarrival_mean if is_hot else cold_interarrival_mean
        lifetime_mean = hot_lifetime_mean if is_hot else cold_lifetime_mean

        current_time = np.random.exponential(interarrival)  # first appearance

        while current_time < num_steps:
            # Create a new "session" for this flow
            duration = max(1.0, np.random.exponential(lifetime_mean))

            # Traffic volume during this session (bytes per second * duration, heavy tailed)
            bps = 100 + int(np.random.pareto(bytes_per_second_pareto_alpha) * 500)
            total_bytes_this_session = int(bps * duration)
            total_pkts_this_session = max(1, total_bytes_this_session // 800)

            if fid not in traces:
                traces[fid] = FlowTrace(flow_id=fid, first_seen=current_time)

            ft = traces[fid]
            ft.arrivals.append(current_time)
            ft.total_bytes += total_bytes_this_session
            ft.total_packets += total_pkts_this_session
            ft.last_seen = max(ft.last_seen, current_time + duration)

            # Next appearance for this flow (after it goes idle)
            idle_time = np.random.exponential(interarrival * (0.8 if is_hot else 1.5))
            current_time += duration + idle_time

    # Sort arrivals for each flow (they may not be perfectly ordered due to randomness)
    for ft in traces.values():
        ft.arrivals.sort()

    return traces


# ------------------------------------------------------------------
# Demo / Quick Validation
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Improved Synthetic Trace Generator Demo ===\n")

    # Create a minimal fake universe similar to the real environment
    fake_universe = [
        {'flow_id': i, 'is_hot': i < 60} for i in range(300)
    ]

    traces = generate_synthetic_trace_from_universe(
        fake_universe,
        num_steps=6000,
        seed=123
    )

    print(f"Generated trace with {len(traces)} unique flows")
    hot_flows = [f for f in traces.values() if f.flow_id < 60]
    cold_flows = [f for f in traces.values() if f.flow_id >= 60]

    if hot_flows:
        avg_arrivals_hot = sum(len(f.arrivals) for f in hot_flows) / len(hot_flows)
        print(f"  Average arrivals per hot flow : {avg_arrivals_hot:.1f}")
    if cold_flows:
        avg_arrivals_cold = sum(len(f.arrivals) for f in cold_flows) / len(cold_flows)
        print(f"  Average arrivals per cold flow: {avg_arrivals_cold:.1f}")

    # Quick simulator test
    sim = TraceSimulator(traces)
    penalty, bytes_after = sim.compute_miss_penalty(5, 1500.0, window=50)
    print(f"\nExample penalty for evicting flow 5 at t=1500: {penalty:.3f} (bytes_after={bytes_after})")

    print("\nSynthetic trace generation is now significantly more realistic.")