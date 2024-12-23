from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
import torch
import numpy as np
from collections import deque
import time
import colorlog
import subprocess
from flow_management_v3 import QNetwork
import os

class RLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RLController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow_table = deque(maxlen=100)  # Track flow entries with max size 100
        self.max_flows = 100
        
        # Initialize DQN model with error handling
        try:
            # Check if CUDA is available, but default to CPU for safety
            self.device = torch.device("cpu")  # Force CPU usage initially
            self.logger.info(f"Using device: {self.device}")
            
            self.state_size = self.max_flows * 4  # 4 features per flow
            self.action_size = 4  # Same as in training
            
            # Initialize model
            self.model = QNetwork(self.state_size, self.action_size).to(self.device)
            
            # Load model with error handling
            model_path = 'models/model_episode_1000.pt'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.logger.info("Successfully loaded DQN model")
            else:
                self.logger.error(f"Model file not found: {model_path}")
                self.model = None
        except Exception as e:
            self.logger.error(f"Failed to initialize RL components: {e}")
            self.model = None
            
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

    def get_flow_stats(self, flow_match):
        """Get flow statistics using ovs-ofctl"""
        try:
            cmd = "sudo ovs-ofctl dump-flows s1"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    # Parse flow entry
                    stats = {
                        'priority': 0,
                        'timeout': 0,
                        'packet_count': 0,
                        'bytes_count': 0
                    }
                    
                    # Extract values from flow entry
                    parts = line.split(',')
                    for part in parts:
                        if 'priority=' in part:
                            stats['priority'] = int(part.split('=')[1])
                        elif 'n_packets=' in part:
                            stats['packet_count'] = int(part.split('=')[1])
                        elif 'n_bytes=' in part:
                            stats['bytes_count'] = int(part.split('=')[1])
                        elif 'idle_timeout=' in part:
                            stats['timeout'] = int(part.split('=')[1])
                    
                    return stats
                    
        except Exception as e:
            self.logger.error(f"Error getting flow stats: {e}")
            return None

    def get_state(self):
        """Preprocess current flow table state for DQN input"""
        flow_info = []
        
        # Get stats for all flows
        flow_stats = []
        for flow in self.flow_table:
            stats = self.get_flow_stats(flow['match'])
            if stats:
                flow_stats.append(stats)
        
        if not flow_stats:
            return np.zeros(self.state_size, dtype=np.float32)
        
        # Normalize features
        max_priority = max(stat['priority'] for stat in flow_stats)
        max_timeout = max(stat['timeout'] for stat in flow_stats)
        max_packets = max(stat['packet_count'] for stat in flow_stats)
        max_bytes = max(stat['bytes_count'] for stat in flow_stats)
        
        # Create normalized state vector
        for stat in flow_stats:
            flow_info.extend([
                stat['priority'] / (max_priority + 1e-6),
                stat['timeout'] / (max_timeout + 1e-6),
                stat['packet_count'] / (max_packets + 1e-6),
                stat['bytes_count'] / (max_bytes + 1e-6)
            ])
        
        # Pad if necessary
        while len(flow_info) < self.state_size:
            flow_info.extend([0, 0, 0, 0])
        
        return np.array(flow_info, dtype=np.float32)

    def select_flow_to_remove(self):
        """Use DQN to select which flow to remove"""
        try:
            if self.model is None:
                self.logger.warning("Model not loaded, using fallback FIFO strategy")
                return 0  # Return first flow (FIFO behavior)
                
            state = self.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.model(state_tensor).argmax().item()
            
            # Map action to flow selection criteria
            if action == 0:  # Lowest priority
                return min(enumerate(self.flow_table), 
                         key=lambda x: self.get_flow_stats(x[1]['match'])['priority'])[0]
            elif action == 1:  # Highest age (timeout)
                return max(enumerate(self.flow_table), 
                         key=lambda x: self.get_flow_stats(x[1]['match'])['timeout'])[0]
            elif action == 2:  # Lowest packet count
                return min(enumerate(self.flow_table), 
                         key=lambda x: self.get_flow_stats(x[1]['match'])['packet_count'])[0]
            else:  # Lowest byte count
                return min(enumerate(self.flow_table), 
                         key=lambda x: self.get_flow_stats(x[1]['match'])['bytes_count'])[0]
                
        except Exception as e:
            self.logger.error(f"Error in flow selection: {e}")
            return 0  # Fallback to FIFO behavior

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

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                           actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                  priority=priority, match=match,
                                  instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                  match=match, instructions=inst)
        datapath.send_msg(mod)

    def remove_flow(self, datapath, match):
        """Remove a specific flow entry using ovs-ofctl"""
        try:
            # Get match fields
            match_fields = match.to_jsondict()['OFPMatch']['oxm_fields']
            
            # Skip if match is empty (table-miss entry)
            if not match_fields:
                self.logger.warning("Attempted to remove table-miss entry, skipping...")
                return

            # Build ovs-ofctl command
            match_str = []
            for field in match_fields:
                field_name = field['OXMTlv']['field']
                field_value = field['OXMTlv']['value']
                
                if field_name == 'in_port':
                    match_str.append(f"in_port={field_value}")
                elif field_name == 'eth_dst':
                    match_str.append(f"dl_dst={field_value}")
                elif field_name == 'eth_src':
                    match_str.append(f"dl_src={field_value}")

            match_criteria = ",".join(match_str)
            
            cmd = f"sudo ovs-ofctl del-flows s1 {match_criteria}"
            self.logger.info(f"Executing command: {cmd}")
            
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully removed flow: {match_criteria}")
            else:
                self.logger.error(f"Failed to remove flow: {result.stderr}")

        except Exception as e:
            self.logger.error(f"Error removing flow: {str(e)}")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Ignore LLDP packets
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src

        # Create match
        match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]

        # If flow table is full, use DQN to select flow to remove
        if len(self.flow_table) >= self.max_flows:
            flow_index = self.select_flow_to_remove()
            removed_flow = self.flow_table[flow_index]
            self.remove_flow(datapath, removed_flow['match'])
            self.flow_table.pop(flow_index)

    @set_ev_cls(ofp_event.EventOFPFlowRemoved, MAIN_DISPATCHER)
    def flow_removed_handler(self, ev):
        msg = ev.msg
        match = msg.match
        self.logger.info(
            f"Flow removed from switch: {match}",
            extra={'color': 'yellow'}
        )