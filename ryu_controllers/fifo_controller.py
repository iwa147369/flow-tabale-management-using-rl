from base_controller import BaseController
from collections import deque
import time
import colorlog

class FIFOController(BaseController):
    def __init__(self, *args, **kwargs):
        super(FIFOController, self).__init__(*args, **kwargs)
        self.flow_table = deque(maxlen=100)
        self.max_flows = 100
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

    def add_flow(self, datapath, priority, match, actions, timeout=0):
        # Log current table status
        self.logger.info(
            f"Current flow table size: {len(self.flow_table)}/{self.max_flows}",
            extra={'color': 'cyan'}
        )

        # Check if flow table is full
        if len(self.flow_table) >= self.max_flows:
            oldest_flow = self.flow_table.popleft()
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
        
        # Log new flow addition
        self.logger.info(
            f"Adding new flow - Priority: {priority}, Match: {match}",
            extra={'color': 'green'}
        )
        
        # Add the new flow to the switch
        super().add_flow(datapath, priority, match, actions, timeout)

    def _packet_in_handler(self, ev):
        # Log packet arrival
        self.logger.debug(
            "Processing new packet...",
            extra={'color': 'blue'}
        )
        super()._packet_in_handler(ev)