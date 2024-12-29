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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
            
            self.state_size = 400  # Fixed input size for the model
            self.action_size = 5  # Same as in training
            
            # Initialize model
            self.model = QNetwork(self.state_size, self.action_size).to(self.device)
            
            # Load model with error handling
            model_path = 'models/model_episode_100.pt'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
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

    def get_state(self):
        """Get and preprocess current flow state directly from switch"""
        try:
            # Get all flows from switch
            cmd = "sudo ovs-ofctl dump-flows s1"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            
            # Clear and update flow table
            self.flow_table.clear()
            
            flow_stats = []
            for line in result.stdout.split('\n'):
                if not line.strip() or 'NXST_FLOW' in line:  # Skip empty lines and header
                    continue
                    
                # Parse flow entry
                stats = {
                    'priority': 0,
                    'timeout': 0,
                    'packet_count': 0,
                    'bytes_count': 0
                }
                
                # Extract values from flow entry
                parts = line.split(',')
                match = None
                actions = None
                
                for part in parts:
                    part = part.strip()
                    if 'priority=' in part:
                        try:
                            stats['priority'] = int(part.split('=')[1])
                        except (IndexError, ValueError):
                            continue
                    elif 'n_packets=' in part:
                        try:
                            stats['packet_count'] = int(part.split('=')[1])
                        except (IndexError, ValueError):
                            continue
                    elif 'n_bytes=' in part:
                        try:
                            stats['bytes_count'] = int(part.split('=')[1])
                        except (IndexError, ValueError):
                            continue
                    elif 'idle_timeout=' in part:
                        try:
                            stats['timeout'] = int(part.split('=')[1])
                        except (IndexError, ValueError):
                            continue
                    elif 'actions=' in part:
                        actions = part.split('=')[1]
                    else:
                        # Assume any other field is part of match
                        if match is None:
                            match = part.strip()
                        else:
                            match += ',' + part.strip()
                
                # Add to flow table if we have both match and actions
                if match and actions:
                    self.flow_table.append({
                        'match': match,
                        'actions': actions,
                        'stats': stats
                    })
                    flow_stats.append(stats)

            if not flow_stats:
                return np.zeros(self.state_size, dtype=np.float32)

            # Normalize features
            max_priority = max(stat['priority'] for stat in flow_stats)
            max_timeout = max(stat['timeout'] for stat in flow_stats)
            max_packets = max(stat['packet_count'] for stat in flow_stats)
            max_bytes = max(stat['bytes_count'] for stat in flow_stats)

            # Create normalized state vector
            flow_info = []
            for stat in flow_stats:
                flow_info.extend([
                    stat['priority'] / (max_priority + 1e-6),
                    stat['timeout'] / (max_timeout + 1e-6),
                    stat['packet_count'] / (max_packets + 1e-6),
                    stat['bytes_count'] / (max_bytes + 1e-6)
                ])

            # Pad or truncate to fixed size
            if len(flow_info) < self.state_size:
                flow_info.extend([0] * (self.state_size - len(flow_info)))
            else:
                flow_info = flow_info[:self.state_size]

            return np.array(flow_info, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Error getting flow state: {e}")
            return np.zeros(self.state_size, dtype=np.float32)

    def select_flow_to_remove(self):
        """Use DQN to select which flow to remove when overflow occurs"""
        start_time = time.time()
        try:
            if self.model is None:
                self.logger.warning("Model not loaded, using fallback FIFO strategy")
                return 0

            state = self.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.model(state_tensor).argmax().item()
            
            # Log timing before action mapping
            inference_time = time.time() - start_time
            self.logger.debug(f"DQN inference took {inference_time:.3f} seconds")
            
            action_start = time.time()
            # Map action to flow selection criteria
            if action == 0:  # Lowest priority
                index = min(enumerate(self.flow_table), 
                         key=lambda x: x[1]['stats']['priority'])[0]
            elif action == 1:  # Highest age (timeout)
                index = max(enumerate(self.flow_table), 
                         key=lambda x: x[1]['stats']['timeout'])[0]
            elif action == 2:  # Lowest packet count
                index = min(enumerate(self.flow_table), 
                         key=lambda x: x[1]['stats']['packet_count'])[0]
            else:  # Lowest byte count
                index = min(enumerate(self.flow_table), 
                         key=lambda x: x[1]['stats']['bytes_count'])[0]
            
            self.logger.debug(f"Action mapping took {time.time() - action_start:.3f} seconds")
            return index
                
        except Exception as e:
            self.logger.error(f"Error in flow selection: {e}")
            return 0

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

    def remove_flow(self, datapath, match_str):
        """Remove a specific flow entry using ovs-ofctl"""
        start_time = time.time()
        try:
            # Extract only the match criteria without metadata
            match_parts = match_str.split(',')
            clean_match = []
            for part in match_parts:
                if any(field in part for field in ['in_port=', 'dl_src=', 'dl_dst=']):
                    clean_match.append(part.strip())
            
            clean_match_str = ','.join(clean_match)
            cmd = f"sudo ovs-ofctl del-flows s1 {clean_match_str}"
            self.logger.info(f"Executing command: {cmd}")
            
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            
            self.logger.debug(f"Flow removal took {time.time() - start_time:.3f} seconds")
            
            if result.returncode == 0:
                self.logger.info(f"Successfully removed flow: {clean_match_str}")
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
        self.logger.info(f"Packet received: {eth.ethertype}")

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        # Learn MAC addresses to avoid flooding
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install flow entry only if not flooding
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            match_str = f"in_port={in_port},dl_dst={dst},dl_src={src}"
            
            # Check actual flow count in switch
            result = subprocess.run("sudo ovs-ofctl dump-flows s1 | wc -l", 
                                 shell=True, capture_output=True, text=True)
            flow_count = int(result.stdout.strip())
            
            # Check if flow table is full
            if flow_count >= self.max_flows:
                flow_index = self.select_flow_to_remove()
                removed_flow = self.flow_table[flow_index]
                self.remove_flow(datapath, removed_flow['match'])
                del self.flow_table[flow_index]
            
            # Add the new flow
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
            else:
                self.add_flow(datapath, 1, match, actions)
                
            # Track the new flow
            self.flow_table.append({
                'match': match_str,
                'actions': 'output:' + str(out_port),
                'stats': {
                    'priority': 1,
                    'timeout': 0,
                    'packet_count': 0,
                    'bytes_count': 0
                }
            })

        # Send packet out
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