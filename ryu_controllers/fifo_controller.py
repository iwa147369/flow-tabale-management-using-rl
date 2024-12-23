from base_controller import BaseController
from collections import deque
import time

class FIFOController(BaseController):
    def __init__(self, *args, **kwargs):
        super(FIFOController, self).__init__(*args, **kwargs)
        self.flow_table = deque(maxlen=100)  # FIFO queue with max size

    def add_flow(self, datapath, priority, match, actions, timeout=0):
        # If flow table is full, remove oldest entry (FIFO)
        if len(self.flow_table) >= self.flow_table.maxlen:
            oldest_flow = self.flow_table.popleft()
            self.remove_flow(datapath, oldest_flow['match'])
            
        # Add new flow to table
        self.flow_table.append({
            'match': match,
            'priority': priority,
            'time': time.time()
        })
        
        super().add_flow(datapath, priority, match, actions, timeout)