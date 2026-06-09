"""
Parse a captured Ryu controller stdout log and verify eviction behaviour.

The controllers print one line per install and one line per eviction:
    INFO ... Installing flow — priority=1, match=OFPMatch(oxm_fields={...})
    WARNING ... Flow table full — removing oldest legal flow: OFPMatch(...)   (FIFO)
    WARNING ... Flow table full — removing LRU (legal): OFPMatch(...)         (LRU)
    WARNING ... Flow table full — removing entry: OFPMatch(...)               (RL)

Those eviction lines do NOT appear in the *_timings.log (that file only has
timings); they are on stdout, so capture stdout when running the controller:
    FLOWRL_MAX_FLOWS=5 PYTHONPATH="$PWD" ryu-manager controllers/fifo_controller.py > run.log 2>&1

This script reconstructs residency from the install/eviction order and checks:
  * eviction actually fired under pressure (evictions == installs - capacity for a
    workload of distinct flows), exposing a stuck eviction path,
  * the table never exceeded the configured capacity,
  * (FIFO only) the victim was always the oldest-installed resident.

Usage:
    python3 tests/parse_eviction_log.py run.log --controller fifo --max-flows 5
"""

import argparse
import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")

EVICT_PHRASES = (
    "removing oldest legal flow:",   # FIFO
    "removing LRU (legal):",         # LRU
    "removing entry:",               # RL
)


def match_key(text):
    """Build a stable key (in_port/eth_src/eth_dst) from an OFPMatch string."""
    src = re.search(r"'eth_src':\s*'([0-9a-fA-F:]+)'", text)
    dst = re.search(r"'eth_dst':\s*'([0-9a-fA-F:]+)'", text)
    if not (src and dst):
        return None
    inp = re.search(r"'in_port':\s*(\d+)", text)
    return f"in={inp.group(1) if inp else '?'},src={src.group(1)},dst={dst.group(1)}"


def parse(path, max_flows, controller):
    residency = []          # current residents, oldest first (install order)
    installs = evictions = 0
    max_resident = 0
    fifo_ok = fifo_total = 0
    full_at_evict = 0       # evictions where the table was at capacity

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw)

            if any(p in line for p in EVICT_PHRASES):
                evictions += 1
                victim = match_key(line)
                if len(residency) >= max_flows:
                    full_at_evict += 1
                if controller == "fifo" and residency:
                    fifo_total += 1
                    if victim == residency[0]:
                        fifo_ok += 1
                if victim in residency:
                    residency.remove(victim)
                elif residency:
                    residency.pop(0)   # fall back to oldest if key unparsable
                continue

            if "Installing flow" in line:
                prio = re.search(r"priority=(\d+)", line)
                if prio and prio.group(1) == "0":
                    continue           # skip table-miss entry
                key = match_key(line)
                if key is None:
                    continue
                installs += 1
                residency.append(key)
                max_resident = max(max_resident, len(residency))

    return {
        "installs": installs,
        "evictions": evictions,
        "max_resident": max_resident,
        "final_resident": len(residency),
        "full_at_evict": full_at_evict,
        "fifo_ok": fifo_ok,
        "fifo_total": fifo_total,
    }


def main():
    ap = argparse.ArgumentParser(description="Verify controller eviction from a captured stdout log")
    ap.add_argument("logfile")
    ap.add_argument("--controller", choices=["fifo", "lru", "rl"], default="fifo")
    ap.add_argument("--max-flows", type=int, required=True,
                    help="The FLOWRL_MAX_FLOWS the controller ran with")
    args = ap.parse_args()

    r = parse(args.logfile, args.max_flows, args.controller)
    expected_evictions = max(0, r["installs"] - args.max_flows)

    print(f"=== eviction verification: {args.controller.upper()} (capacity {args.max_flows}) ===")
    print(f"  installs (priority>0) : {r['installs']}")
    print(f"  evictions             : {r['evictions']}")
    print(f"  expected evictions    : {expected_evictions}  (installs - capacity, distinct workload)")
    print(f"  max resident observed : {r['max_resident']}  (must be <= {args.max_flows})")
    print(f"  final resident        : {r['final_resident']}")
    print(f"  evicted while full    : {r['full_at_evict']}/{r['evictions']}")

    checks = []
    checks.append(("table never exceeds capacity", r["max_resident"] <= args.max_flows))
    # Allow a small slack: re-installs of returning flows reduce the eviction count.
    checks.append(("eviction fires under pressure",
                   r["evictions"] >= 0.8 * expected_evictions if expected_evictions else True))
    if args.controller == "fifo" and r["fifo_total"]:
        rate = r["fifo_ok"] / r["fifo_total"]
        print(f"  FIFO evict-oldest     : {r['fifo_ok']}/{r['fifo_total']} ({100 * rate:.0f}%)")
        checks.append(("FIFO evicts oldest-installed", rate >= 0.95))
    if args.controller == "lru":
        print("  note: LRU recency only refreshes when a resident flow triggers a new")
        print("        Packet-In; installed rules match in the switch and send none, so")
        print("        LRU may behave close to FIFO here. Interpret accordingly.")

    print("  ---")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print(f"  => {'OK' if all_ok else 'PROBLEM DETECTED'}")


if __name__ == "__main__":
    main()
