import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
import torch
import numpy as np
import time
import colorlog
import datetime
import os

from training.model import QNetwork
from controllers.flow_utils import is_critical_flow, filter_evictable_flows
from training.trace_simulator import TraceRecorder


class RLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RLController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}

        # Unified flow table: list of dicts with consistent schema.
        # Each entry: {match, priority, timeout (age in s), packet_count, bytes_count}
        self.flow_table = []
        self.max_flows = 100
        self.datapath = None
        self.log_file = "rl_timings.log"

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            self.state_size = 400
            self.action_size = 4

            self.model = QNetwork(self.state_size, self.action_size).to(self.device)

            model_path = 'models/model_episode_1000.pt'
            if os.path.exists(model_path):
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=True)
                )
                self.model.eval()
                self.logger.info("Successfully loaded DQN model")
            else:
                self.logger.error(f"Model file not found: {model_path}")
                self.model = None
        except Exception as e:
            self.logger.error(f"Failed to initialize RL components: {e}")
            self.model = None

        # --- Trace recording (Phase 0.5) ---
        self.recording_enabled = os.environ.get("FLOWRL_RECORD_TRACE", "0") == "1"
        self.trace_recorder = None
        if self.recording_enabled:
            self.trace_recorder = TraceRecorder()
            self.logger.info("Trace recording ENABLED (set FLOWRL_RECORD_TRACE=0 to disable)")

        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)s:%(name)s:%(message)s',
            log_colors={
                'DEBUG': 'cyan', 'INFO': 'green',
                'WARNING': 'yellow', 'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        self.logger.handlers = [handler]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def log_timing(self, action, duration):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {action}: {duration:.5f} seconds\n")

    def _match_key(self, match):
        """Stable string key for an OFPMatch object."""
        try:
            fields = match.to_jsondict()['OFPMatch']['oxm_fields']
            return str(sorted(str(f) for f in fields))
        except Exception:
            return str(match)

    def request_flow_stats(self, datapath):
        """Ask the switch for fresh per-flow statistics."""
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # ── OpenFlow event handlers ───────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapath = datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        # Install table-miss without tracking it in flow_table
        self._install_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Update local flow table stats from the switch reply."""
        for stat in ev.msg.body:
            if stat.priority == 0:   # skip table-miss
                continue
            key = self._match_key(stat.match)
            for entry in self.flow_table:
                if entry['match_key'] == key:
                    entry['timeout'] = stat.duration_sec
                    entry['packet_count'] = stat.packet_count
                    entry['bytes_count'] = stat.byte_count
                    break

            # Trace recording: update byte/packet counts for real traffic analysis
            if self.trace_recorder is not None:
                flow_id = hash(key) & 0xFFFFFFFF
                self.trace_recorder.record_flow(
                    flow_id,
                    bytes=stat.byte_count,
                    packets=stat.packet_count
                )

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        match_key = self._match_key(ev.msg.match)
        self.flow_table = [e for e in self.flow_table if e['match_key'] != match_key]
        self.logger.info(f"Flow removed from switch: {ev.msg.match}")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst, src = eth.dst, eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    # ── Flow table management ─────────────────────────────────────────────────

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        start_time = time.time()

        match_key = self._match_key(match)

        # Skip if already tracked
        if any(e['match_key'] == match_key for e in self.flow_table):
            self.logger.info("Flow already exists, skipping addition")
            return

        # Evict if full — but only from the legal (non-critical) candidate set
        if len(self.flow_table) >= self.max_flows:
            idx = self.select_flow_to_remove()
            if idx < len(self.flow_table):
                evicted = self.flow_table.pop(idx)
                self.logger.warning(f"Flow table full — removing entry: {evicted.get('match')}")
                self.remove_flow(datapath, evicted['match'], evicted.get('priority', 1))
            else:
                self.logger.error("select_flow_to_remove returned invalid index — skipping eviction")

        # Track locally
        self.flow_table.append({
            'match':         match,
            'match_key':     match_key,
            'priority':      priority,
            'timeout':       0,
            'packet_count':  0,
            'bytes_count':   0,
        })

        self._install_flow(datapath, priority, match, actions, buffer_id)
        self.log_timing("Install Flow", time.time() - start_time)

        # Trace recording (Phase 0.5)
        if self.trace_recorder is not None:
            flow_id = hash(self._match_key(match)) & 0xFFFFFFFF  # stable int id
            self.trace_recorder.record_flow(flow_id, bytes=0, packets=0)

        # Request a stats refresh so the next eviction decision has fresh data
        self.request_flow_stats(datapath)

    def remove_flow(self, datapath, match, priority):
        """Delete a flow from the switch via OFPFlowMod (no SSH needed)."""
        start_time = time.time()
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            priority=priority,
            match=match,
        )
        datapath.send_msg(mod)
        self.log_timing("Remove Flow", time.time() - start_time)

    def _install_flow(self, datapath, priority, match, actions, buffer_id=None):
        """Send OFPFlowMod ADD to install a rule on the switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        kwargs = dict(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            hard_timeout=0,
            flags=ofproto.OFPFF_SEND_FLOW_REM,
        )
        if buffer_id and buffer_id != ofproto.OFP_NO_BUFFER:
            kwargs['buffer_id'] = buffer_id

        datapath.send_msg(parser.OFPFlowMod(**kwargs))
        self.logger.info(f"Installing flow — priority={priority}, match={match}")

    # ── RL inference ──────────────────────────────────────────────────────────

    def get_state(self, use_evictable_only: bool = True):
        """
        Build 400-dim state vector from local flow table.

        When use_evictable_only=True (default), only non-critical flows are
        included. This ensures the RL agent (and any code using this state)
        never sees flows it is not allowed to evict.
        """
        flows = self.flow_table
        if use_evictable_only:
            flows = filter_evictable_flows(self.flow_table)

        if not flows:
            return np.zeros(self.state_size, dtype=np.float32)

        max_priority = max(e.get('priority', 0)      for e in flows) + 1e-6
        max_timeout  = max(e.get('timeout', 0)       for e in flows) + 1e-6
        max_packets  = max(e.get('packet_count', 0)  for e in flows) + 1e-6
        max_bytes    = max(e.get('bytes_count', 0)   for e in flows) + 1e-6

        flow_info = []
        for e in flows:
            flow_info.extend([
                e.get('priority', 0)     / max_priority,
                e.get('timeout', 0)      / max_timeout,
                e.get('packet_count', 0) / max_packets,
                e.get('bytes_count', 0)  / max_bytes,
            ])

        # Pad or truncate to fixed size
        if len(flow_info) < self.state_size:
            flow_info.extend([0.0] * (self.state_size - len(flow_info)))
        else:
            flow_info = flow_info[:self.state_size]

        return np.array(flow_info, dtype=np.float32)

    def select_flow_to_remove(self):
        """Use DQN to choose which flow to evict, but never touch critical flows."""
        start_time = time.time()

        if not self.flow_table:
            self.log_timing("Remove Flow Decision", time.time() - start_time)
            return 0

        # Hard pre-filter: the agent is only allowed to consider non-critical flows
        evictable = filter_evictable_flows(self.flow_table)

        if not evictable:
            # Every remaining flow is critical (very rare). Fall back to safest choice.
            self.logger.warning("All flows in table are critical — refusing to evict")
            self.log_timing("Remove Flow Decision", time.time() - start_time)
            return 0

        if self.model is None:
            # Model not available → fall back to safest heuristic on evictable flows only
            idx = self._fallback_evict_on_evictable(evictable)
            self.log_timing("Remove Flow Decision", time.time() - start_time)
            return self._index_in_original_table(idx, evictable)

        try:
            # Build state ONLY from evictable flows so the model never sees protected ones
            state = self.get_state(use_evictable_only=True)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.model(state_tensor).argmax().item()

            if action == 0:
                idx = min(range(len(evictable)),
                          key=lambda i: evictable[i]['priority'])
            elif action == 1:
                idx = max(range(len(evictable)),
                          key=lambda i: evictable[i].get('timeout', 0))
            elif action == 2:
                idx = min(range(len(evictable)),
                          key=lambda i: evictable[i].get('packet_count', 0))
            else:
                idx = min(range(len(evictable)),
                          key=lambda i: evictable[i].get('bytes_count', 0))

            chosen = evictable[idx]
            self.logger.debug(f"DQN chose action {action} → evict flow (prio={chosen.get('priority')})")
            self.log_timing("Remove Flow Decision", time.time() - start_time)
            return self._index_in_original_table(chosen, self.flow_table)

        except Exception as e:
            self.logger.error(f"Error in flow selection: {e}, falling back safely")
            idx = self._fallback_evict_on_evictable(evictable)
            self.log_timing("Remove Flow Decision", time.time() - start_time)
            return self._index_in_original_table(idx, evictable)

    # ── Helpers for critical flow aware eviction ─────────────────────────────

    def _fallback_evict_on_evictable(self, evictable: list):
        """Safe fallback: evict lowest priority among legal candidates."""
        if not evictable:
            return 0
        return min(range(len(evictable)), key=lambda i: evictable[i].get('priority', 999))

    def _index_in_original_table(self, target, table):
        """Given a flow (or index in a filtered list), return its index in the original table."""
        if isinstance(target, int):
            target = table[target] if target < len(table) else table[0]

        for i, f in enumerate(table):
            if f.get('match_key') == target.get('match_key'):
                return i
            if f.get('match') == target.get('match'):
                return i
        return 0  # last resort

    # ── Trace recording helpers (Phase 0.5) ─────────────────────────────────

    def save_trace(self, path: str = None):
        """Save the collected trace (if recording was enabled)."""
        if self.trace_recorder is None:
            self.logger.warning("Trace recording was not enabled. Set FLOWRL_RECORD_TRACE=1 before starting ryu-manager.")
            return

        if path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"traces/trace_rl_{timestamp}.pkl"

        # Ensure traces directory exists
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self.trace_recorder.save(path)
        self.logger.info(f"Trace saved to {path}")

    def __del__(self):
        """Auto-save trace on shutdown if recording was enabled."""
        if getattr(self, "trace_recorder", None) is not None:
            try:
                self.save_trace()
            except Exception:
                pass  # Best effort during shutdown
