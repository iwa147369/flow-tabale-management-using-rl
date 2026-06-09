from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
import time
import colorlog
import datetime

from controllers.flow_utils import is_critical_flow, filter_evictable_flows, flow_id_of
from training.trace_simulator import TraceRecorder
import os
import datetime


class FIFOController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(FIFOController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow_table = []
        self.max_flows = 100
        self.log_file = "fifo_timings.log"

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
        self.logger.info(f"Initialized FIFO Controller (max {self.max_flows} flows)")

        # --- Trace recording (Phase 0.5) ---
        self.recording_enabled = os.environ.get("FLOWRL_RECORD_TRACE", "0") == "1"
        self.trace_recorder = None
        if self.recording_enabled:
            self.trace_recorder = TraceRecorder()
            self.logger.info("Trace recording ENABLED (set FLOWRL_RECORD_TRACE=0 to disable)")

    def log_timing(self, action, duration):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {action}: {duration:.5f} seconds\n")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self._install_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        start_time = time.time()

        # Stable identity for dedup + eviction. Entries MUST carry this: the
        # eviction filter below compares by match_key, and without it the filter
        # would wipe the whole table (None != None is False for every entry).
        match_key = flow_id_of(match)
        if any(e.get('match_key') == match_key for e in self.flow_table):
            self.logger.info("Flow already exists, skipping addition")
            return

        if len(self.flow_table) >= self.max_flows:
            evictable = filter_evictable_flows(self.flow_table)
            if evictable:
                # For pure FIFO we still prefer true oldest among legal candidates
                oldest = min(evictable, key=lambda f: f.get('time', 0))
                self.flow_table = [f for f in self.flow_table if f.get('match_key') != oldest.get('match_key')]
                self.logger.warning(f"Flow table full — removing oldest legal flow: {oldest.get('match')}")
                self.remove_flow(datapath, oldest['match'], oldest.get('priority', 1))
            else:
                self.logger.warning("No legal flows to evict — table full of critical entries")

        self.flow_table.append({'match': match, 'match_key': match_key,
                                'priority': priority, 'time': time.time()})
        self._install_flow(datapath, priority, match, actions, buffer_id)
        self.log_timing("Install Flow", time.time() - start_time)

        # Trace recording (Phase 0.5): install = one arrival (bytes captured on removal)
        if self.trace_recorder is not None:
            self.trace_recorder.record_flow(flow_id_of(match), bytes=0, packets=0)

    def remove_flow(self, datapath, match, priority):
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
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        kwargs = dict(datapath=datapath, priority=priority, match=match,
                      instructions=inst, hard_timeout=0,
                      flags=ofproto.OFPFF_SEND_FLOW_REM)
        if buffer_id and buffer_id != ofproto.OFP_NO_BUFFER:
            kwargs['buffer_id'] = buffer_id
        datapath.send_msg(parser.OFPFlowMod(**kwargs))
        self.logger.info(f"Installing flow — priority={priority}, match={match}")

    def save_trace(self, path: str = None):
        """Save the collected trace (if recording was enabled)."""
        if self.trace_recorder is None:
            self.logger.warning("Trace recording was not enabled. Set FLOWRL_RECORD_TRACE=1 before starting ryu-manager.")
            return

        if path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"traces/trace_fifo_{timestamp}.pkl"

        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self.trace_recorder.save(path)
        self.logger.info(f"Trace saved to {path}")

    def __del__(self):
        if getattr(self, "trace_recorder", None) is not None:
            try:
                self.save_trace()
            except Exception:
                pass

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

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        # The FlowRemoved event carries the flow's lifetime byte/packet counts.
        # Record them against the same flow id used at install (no new arrival).
        if self.trace_recorder is not None and ev.msg.priority != 0:
            self.trace_recorder.add_stats(
                flow_id_of(ev.msg.match), ev.msg.byte_count, ev.msg.packet_count
            )
        self.logger.info(f"Flow removed from switch: {ev.msg.match}")
