from ryu.app.simple_switch_13 import SimpleSwitch13
from collections import deque
import time
import colorlog

class FIFOController(SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(FIFOController, self).__init__(*args, **kwargs)
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

    def remove_flow(self, datapath, match):
        """Remove a specific flow entry"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            match=match
        )
        datapath.send_msg(mod)
        self.logger.info(
            f"Removed flow entry: {match}",
            extra={'color': 'red'}
        )

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        self.logger.info(
            f"Current flow table size: {len(self.flow_table)}/{self.max_flows}",
            extra={'color': 'cyan'}
        )

        # Check if flow table is full
        if len(self.flow_table) >= self.max_flows:
            oldest_flow = self.flow_table.popleft()  # Remove oldest flow (FIFO)
            self.logger.warning(
                "Flow table full! Removing oldest entry...",
                extra={'color': 'yellow', 'bold': True}
            )
            self.remove_flow(datapath, oldest_flow['match'])
            
        # Add new flow to our tracking table
        flow_entry = {
            'match': match,
            'priority': priority,
            'time': time.time()
        }
        self.flow_table.append(flow_entry)
        
        self.logger.info(
            f"Adding new flow - Priority: {priority}, Match: {match}",
            extra={'color': 'green'}
        )
        
        # Add the new flow to the switch using parent's add_flow
        super().add_flow(datapath, priority, match, actions, buffer_id)