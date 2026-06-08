"""
Shared utilities for flow table management across controllers.

Critical flow protection is a hard requirement: certain flows must never be
evicted by the agent or by classical policies, because doing so would make the
switch blind or break control plane communication.
"""

from typing import Any, Dict


def is_critical_flow(flow: Dict[str, Any]) -> bool:
    """
    Return True if this flow entry must never be evicted.

    Current policy (matches the design in docs/memory.md):
    - priority == 0          → OpenFlow table-miss entry (wildcard). Removing it
                               makes the switch unable to send Packet-Ins for
                               new flows.
    - priority >= 65535      → Maximum priority / reserved infrastructure flows
                               (common convention in many controllers).

    Future extensions (recommended):
    - Match overlaps controller IP / controller port
    - Cookie == 0xDEADBEEF or other reserved cookie
    - Specific eth_type (ARP, LLDP) when we want to be extra safe

    The function is deliberately tolerant of different storage schemas used by
    the various controllers (some store only {'match', 'priority'}, the RL
    controller stores the richer stats dict).
    """
    prio = flow.get("priority", None)
    if prio is None:
        # If we don't know the priority, be conservative and do not protect.
        return False

    if prio == 0:
        return True
    if prio >= 65535:
        return True

    # Future: add more sophisticated checks here (cookie, match fields, etc.)
    return False


def filter_evictable_flows(flow_table: list) -> list:
    """
    Return only the flows that are legal to evict.

    Used by the RL controller before building state and before calling the model,
    and by FIFO/LRU when they need to pick a victim.
    """
    return [f for f in flow_table if not is_critical_flow(f)]