from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from collections import deque
import time
import colorlog

class FIFOController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(FIFOController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow_table = deque(maxlen=100)  # Track flow entries with FIFO queue
        self.max_flows = 100  # Maximum number of flows allowed
        
        # Set up colored logging
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)s:%(name)s:%(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green', 
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        self.logger.handlers = [handler]
        
        self.logger.info(
            f"Initialized FIFO Controller with max {self.max_flows} flows",
            extra={'color': 'green', 'bold': True}
        )

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def remove_flow(self, datapath, match):
        """Remove a specific flow entry"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Skip removal if match is empty (table-miss entry)
        if not match.to_jsondict()['OFPMatch']['oxm_fields']:
            self.logger.warning("Attempted to remove table-miss entry, skipping...")
            return

        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE_STRICT,  # Use STRICT to match exactly
            match=match,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            table_id=ofproto.OFPTT_ALL
        )
        datapath.send_msg(mod)
        self.logger.info(
            f"Removed specific flow entry: {match.to_jsondict()['OFPMatch']}",
            extra={'color': 'red'}
        )

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        # First, check if this exact match already exists
        for flow in self.flow_table:
            if flow['match'] == match:
                self.logger.info("Flow already exists, skipping addition")
                return

        # If flow table is full, remove oldest entry (FIFO policy)
        if len(self.flow_table) >= self.max_flows:
            oldest_flow = self.flow_table.popleft()
            self.logger.warning(
                f"Flow table full! Removing oldest entry: {oldest_flow['match']}",
                extra={'color': 'yellow', 'bold': True}
            )
            self.remove_flow(datapath, oldest_flow['match'])

        # Add new flow
        flow_entry = {
            'match': match,
            'priority': priority,
            'time': time.time()
        }
        self.flow_table.append(flow_entry)
        
        # Install flow in switch
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                instructions=inst,
                hard_timeout=0,  # Flow entry never expires
                flags=ofproto.OFPFF_SEND_FLOW_REM  # Request flow removal notification
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=inst,
                hard_timeout=0,
                flags=ofproto.OFPFF_SEND_FLOW_REM
            )
        
        self.logger.info(
            f"Installing new flow - Priority: {priority}, Match: {match}",
            extra={'color': 'green'}
        )
        datapath.send_msg(mod)

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
            # ignore lldp packet
            return
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        msg = ev.msg
        match = msg.match
        self.logger.info(
            f"Flow removed from switch: {match}",
            extra={'color': 'yellow'}
        )