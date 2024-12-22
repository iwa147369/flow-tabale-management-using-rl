from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
import torch
import numpy as np
from flow_management_v3 import DoubleDQNAgent, FlowTableEnvironment, TABLE_SIZE

class FlowManagerController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(FlowManagerController, self).__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize environment and agent
        self.env = FlowTableEnvironment()
        self.agent = DoubleDQNAgent(self.env.observation_space.shape[0], self.env.action_space.n)
        
        # Load the trained model
        self.load_trained_model()
        
        # Initialize flow table
        self.flow_table = []
        self.mac_to_port = {}

    def load_trained_model(self, model_path='models/model_episode_500.pt'):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.agent.q_network.load_state_dict(checkpoint['q_network'])
            self.agent.target_q_network.load_state_dict(checkpoint['target_q_network'])
            self.logger.info("Loaded trained model successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

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

    def add_flow(self, datapath, priority, match, actions, timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                           actions)]
        
        mod = parser.OFPFlowMod(datapath=datapath,
                               priority=priority,
                               match=match,
                               instructions=inst,
                               hard_timeout=timeout)
        datapath.send_msg(mod)

    def remove_flow(self, datapath, flow_index):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        if 0 <= flow_index < len(self.flow_table):
            flow = self.flow_table[flow_index]
            match = parser.OFPMatch(**flow['match'])
            
            mod = parser.OFPFlowMod(
                datapath=datapath,
                command=ofproto.OFPFC_DELETE,
                out_port=ofproto.OFPP_ANY,
                out_group=ofproto.OFPG_ANY,
                match=match
            )
            datapath.send_msg(mod)
            self.flow_table.pop(flow_index)

    def get_state(self):
        if len(self.flow_table) < TABLE_SIZE:
            return None

        flow_info = []
        max_priority = max(flow['priority'] for flow in self.flow_table)
        max_timeout = max(flow['timeout'] for flow in self.flow_table)
        max_packets = max(flow['packet_count'] for flow in self.flow_table)
        max_bytes = max(flow['bytes_count'] for flow in self.flow_table)

        for flow in self.flow_table:
            flow_info.extend([
                flow['priority'] / max_priority,
                flow['timeout'] / max_timeout,
                flow['packet_count'] / max_packets,
                flow['bytes_count'] / max_bytes
            ])

        return np.array(flow_info, dtype=np.float32)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
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

        # Learn MAC address to avoid FLOOD
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            # Generate flow statistics similar to your training environment
            new_flow = {
                'match': {'in_port': in_port, 'eth_dst': dst, 'eth_src': src},
                'priority': np.random.randint(0, 100),
                'timeout': np.random.randint(0, 100),
                'packet_count': np.random.randint(0, 100),
                'bytes_count': np.random.randint(0, 100),
                'actions': actions
            }

            # Manage flow table using the trained agent
            if len(self.flow_table) >= TABLE_SIZE:
                state = self.get_state()
                if state is not None:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        action = self.agent.q_network(state_tensor).argmax().item()
                        self.remove_flow(datapath, action)

            # Add new flow
            self.flow_table.append(new_flow)
            self.add_flow(datapath, new_flow['priority'], match, actions, 
                         timeout=new_flow['timeout'])

        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                 in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out) 