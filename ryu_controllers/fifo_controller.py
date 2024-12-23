from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from collections import deque
import time
import logging

class FIFOController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(FIFOController, self).__init__(*args, **kwargs)
        self.flow_table = deque(maxlen=100)  # FIFO queue with 100 entry limit
        self.mac_to_port = {}
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug('Initializing FIFO Controller')

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.logger.debug(f'Switch connected: datapath_id={datapath.id}')

        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                        ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.debug('Installed table-miss flow entry')

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.logger.debug(f'Adding flow - Priority: {priority}, Match: {match}, Actions: {actions}')

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                           actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                  priority=priority, match=match,
                                  instructions=inst, hard_timeout=timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                  match=match, instructions=inst,
                                  hard_timeout=timeout)

        # If flow table is full, remove oldest entry (FIFO)
        if len(self.flow_table) >= 100:
            oldest_flow = self.flow_table.popleft()
            self.remove_flow(datapath, oldest_flow['match'])
            self.logger.info(f"FIFO: Table full! Removed flow - Match: {oldest_flow['match']}, "
                           f"Priority: {oldest_flow['priority']}, Added time: {oldest_flow['time']}")
            self.logger.debug(f'Flow table size after removal: {len(self.flow_table)}')

        # Add new flow to our table
        flow_entry = {
            'match': match,
            'priority': priority,
            'actions': actions,
            'time': time.time()
        }
        self.flow_table.append(flow_entry)
        self.logger.info(f"FIFO: Added new flow (table size: {len(self.flow_table)})")
        self.logger.debug(f'Flow entry details: {flow_entry}')

        datapath.send_msg(mod)

    def remove_flow(self, datapath, match):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.logger.debug(f'Removing flow - Match: {match}')

        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=match
        )
        self.logger.info(f"FIFO: Removed flow - Match: {match}")
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

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        self.logger.debug(f'Packet in - dpid: {dpid}, src: {src}, dst: {dst}, in_port: {in_port}')

        # Learn MAC addresses to avoid FLOOD
        self.mac_to_port[dpid][src] = in_port
        self.logger.debug(f'MAC learned - dpid: {dpid}, mac: {src}, port: {in_port}')

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
            self.logger.debug(f'Destination known - sending to port {out_port}')
        else:
            out_port = ofproto.OFPP_FLOOD
            self.logger.debug('Destination unknown - flooding')

        actions = [parser.OFPActionOutput(out_port)]

        # Install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.logger.debug(f'Installing flow - match: {match}, actions: {actions}')
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
            else:
                self.add_flow(datapath, 1, match, actions)

        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
        self.logger.debug('Packet sent out')