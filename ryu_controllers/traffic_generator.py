from scapy.all import *
import random
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class TrafficGenerator:
    def __init__(self, num_hosts=100):
        self.num_hosts = num_hosts
        self.hosts = [f"00:00:00:00:00:{i:02x}" for i in range(num_hosts)]
        
    def generate_packet(self):
        src = random.choice(self.hosts)
        dst = random.choice([h for h in self.hosts if h != src])
        
        # Create Ethernet packet
        pkt = Ether(src=src, dst=dst)
        return pkt

    def generate_traffic_pattern(self, pattern="uniform", duration=300, rate=100):
        """
        Generate traffic following different patterns:
        - uniform: Constant rate
        - bursty: Random bursts of traffic
        - periodic: Sinusoidal traffic pattern
        """
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if pattern == "uniform":
                time.sleep(1/rate)
                yield self.generate_packet()
                
            elif pattern == "bursty":
                if random.random() < 0.1:  # 10% chance of burst
                    # Generate burst of 10-50 packets
                    for _ in range(random.randint(10, 50)):
                        yield self.generate_packet()
                time.sleep(1/rate)
                
            elif pattern == "periodic":
                # Sinusoidal rate variation
                t = time.time() - start_time
                current_rate = rate * (1 + np.sin(2 * np.pi * t / 60))  # 1-minute period
                time.sleep(1/current_rate)
                yield self.generate_packet()

    def send_traffic(self, pattern="uniform", duration=300, rate=100):
        """Send traffic to the network"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            for pkt in self.generate_traffic_pattern(pattern, duration, rate):
                executor.submit(sendp, pkt, verbose=False)

if __name__ == "__main__":
    generator = TrafficGenerator()
    
    # Example: Generate 5 minutes of each traffic pattern
    patterns = ["uniform", "bursty", "periodic"]
    for pattern in patterns:
        print(f"Generating {pattern} traffic...")
        generator.send_traffic(pattern=pattern, duration=300, rate=100) 