from base_controller import BaseController
import time

class LRUController(BaseController):
    def __init__(self, *args, **kwargs):
        super(LRUController, self).__init__(*args, **kwargs)
        self.flow_table = []
        
    def add_flow(self, datapath, priority, match, actions, timeout=0):
        # If flow table is full, remove least recently used entry
        if len(self.flow_table) >= 100:
            lru_flow = min(self.flow_table, key=lambda x: x['last_used'])
            self.flow_table.remove(lru_flow)
            self.remove_flow(datapath, lru_flow['match'])
            
        # Add new flow to table
        self.flow_table.append({
            'match': match,
            'priority': priority,
            'last_used': time.time()
        })
        
        super().add_flow(datapath, priority, match, actions, timeout) 