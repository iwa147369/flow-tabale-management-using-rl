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

We cannot observe the controller's internal table size from the log, so the
reliable signal is the **eviction-to-install ratio**. With more distinct active
flows than the capacity, a correct policy must evict on nearly every install once
the table is warm (ratio → 1). A ratio near 1/capacity means the table keeps
emptying — a broken eviction path (e.g. a filter that wipes the whole table).
For FIFO we additionally check the victim is the oldest-installed resident.

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


def parse(path, controller):
    installs = evictions = 0
    distinct = set()
    order = []               # resident keys in install order (deduped) for FIFO check
    resident = set()
    fifo_ok = fifo_total = 0

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw)

            if any(p in line for p in EVICT_PHRASES):
                evictions += 1
                victim = match_key(line)
                if controller == "fifo" and order:
                    fifo_total += 1
                    if victim == order[0]:
                        fifo_ok += 1
                if victim in resident:
                    resident.discard(victim)
                    order = [k for k in order if k != victim]
                elif order:
                    resident.discard(order.pop(0))
                continue

            if "Installing flow" in line:
                prio = re.search(r"priority=(\d+)", line)
                if prio and prio.group(1) == "0":
                    continue           # skip table-miss entry
                key = match_key(line)
                if key is None:
                    continue
                installs += 1
                distinct.add(key)
                if key not in resident:
                    resident.add(key)
                    order.append(key)

    return {
        "installs": installs,
        "evictions": evictions,
        "distinct": len(distinct),
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

    r = parse(args.logfile, args.controller)
    ratio = r["evictions"] / r["installs"] if r["installs"] else 0.0
    # Once warm, a bounded table evicts on nearly every install: target ratio.
    target = max(0.0, (r["installs"] - args.max_flows) / r["installs"]) if r["installs"] else 0.0

    print(f"=== eviction verification: {args.controller.upper()} (capacity {args.max_flows}) ===")
    print(f"  installs (priority>0)   : {r['installs']}")
    print(f"  distinct flows          : {r['distinct']}  (re-installs imply thrash under a small table)")
    print(f"  evictions               : {r['evictions']}")
    print(f"  evict/install ratio     : {ratio:.3f}   (target ~{target:.3f}; ~1/capacity means the table keeps emptying)")

    checks = []
    checks.append(("controller produced installs (it actually ran)", r["installs"] > 0))
    # Only meaningful once there is real pressure (installs well above capacity).
    if r["installs"] > 3 * args.max_flows:
        checks.append(("eviction keeps pace (table stays bounded)", ratio >= 0.8))
    if args.controller == "fifo" and r["fifo_total"]:
        rate = r["fifo_ok"] / r["fifo_total"]
        print(f"  FIFO evict-oldest       : {r['fifo_ok']}/{r['fifo_total']} ({100 * rate:.0f}%)")
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
