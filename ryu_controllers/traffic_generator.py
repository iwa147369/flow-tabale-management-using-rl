from scapy.all import *
import random
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class TrafficFlow:
    def __init__(self, priority, timeout, flow_type, qos_requirements=None):
        self.priority = priority
        self.timeout = timeout
        self.flow_type = flow_type
        self.qos_requirements = qos_requirements or {}

class TrafficGenerator:
    def __init__(self, num_hosts=100):
        self.num_hosts = num_hosts
        self.hosts = [f"00:00:00:00:00:{i:02x}" for i in range(num_hosts)]
        
        # Define different types of traffic flows with QoS requirements
        self.flow_types = {
            'realtime': TrafficFlow(
                priority=3,
                timeout=5,
                flow_type='realtime',
                qos_requirements={
                    'max_latency': 10,    # ms
                    'min_bandwidth': 10,   # Mbps
                    'packet_loss': 0.001   # 0.1%
                }
            ),
            'streaming': TrafficFlow(
                priority=2,
                timeout=30,
                flow_type='streaming',
                qos_requirements={
                    'max_latency': 100,    # ms
                    'min_bandwidth': 5,     # Mbps
                    'packet_loss': 0.01     # 1%
                }
            ),
            'background': TrafficFlow(
                priority=1,
                timeout=60,
                flow_type='background',
                qos_requirements={
                    'max_latency': 1000,    # ms
                    'min_bandwidth': 1,      # Mbps
                    'packet_loss': 0.05      # 5%
                }
            )
        }
        
    def generate_packet(self, flow_type='background'):
        src = random.choice(self.hosts)
        dst = random.choice([h for h in self.hosts if h != src])
        
        # Create Ethernet packet with flow type in payload
        pkt = Ether(src=src, dst=dst)/IP()/Raw(load=flow_type)
        return pkt

    def generate_traffic_pattern(self, pattern="uniform", duration=300, rate=100):
        """
        Generate mixed traffic following different patterns with varying flow types
        """
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Select flow type based on probabilities
            flow_type = random.choices(
                ['realtime', 'streaming', 'background'],
                weights=[0.2, 0.3, 0.5]  # 20% realtime, 30% streaming, 50% background
            )[0]
            
            if pattern == "uniform":
                time.sleep(1/rate)
                yield self.generate_packet(flow_type)
                
            elif pattern == "bursty":
                if random.random() < 0.1:
                    # Generate burst with same flow type
                    burst_size = random.randint(10, 50)
                    for _ in range(burst_size):
                        yield self.generate_packet(flow_type)
                time.sleep(1/rate)
                
            elif pattern == "periodic":
                t = time.time() - start_time
                current_rate = rate * (1 + np.sin(2 * np.pi * t / 60))
                time.sleep(1/current_rate)
                yield self.generate_packet(flow_type)

    def send_traffic(self, pattern="uniform", duration=300, rate=100):
        """Send traffic to the network"""
        print(f"Generating {pattern} traffic with mixed priorities...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            for pkt in self.generate_traffic_pattern(pattern, duration, rate):
                executor.submit(sendp, pkt, verbose=False)

    def get_flow_info(self, packet):
        """Extract flow type and associated requirements"""
        flow_type = packet[Raw].load.decode()
        return self.flow_types[flow_type]

if __name__ == "__main__":
    generator = TrafficGenerator()
    
    # Test each pattern for 5 minutes
    patterns = ["uniform", "bursty", "periodic"]
    for pattern in patterns:
        print(f"\nStarting {pattern} traffic pattern test...")
        print("Flow distribution: 20% realtime, 30% streaming, 50% background")
        generator.send_traffic(pattern=pattern, duration=300, rate=100) 