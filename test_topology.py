from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
import random
import time
import threading
import os

class TestTopo(Topo):
    def build(self):
        # Add one switch
        s1 = self.addSwitch('s1')
        
        # Add 12 hosts to ensure we can generate over 100 flows
        # 12 hosts can generate up to 12 * 11 = 132 unique flows
        for i in range(12):
            host = self.addHost(f'h{i+1}')
            self.addLink(host, s1)

def generate_all_flows(net):
    """Generate flows between all possible host pairs"""
    hosts = net.hosts
    flow_count = 0
    
    print("Generating flows between all hosts...")
    for h1 in hosts:
        for h2 in hosts:
            if h1 != h2:
                print(f"Generating flow from {h1.name} to {h2.name}")
                h1.cmd(f'ping -c 1 {h2.IP()} &')
                flow_count += 1
                time.sleep(0.1)  # Small delay between pings
                
                # Print current flow count
                current_flows = int(os.popen('ovs-ofctl dump-flows s1 | wc -l').read()) - 1
                print(f"Current flow count: {current_flows}")
                
                # Check if we've exceeded the flow table size
                if current_flows > 100:
                    print("Flow table overflow achieved!")
    
    print(f"Total flows generated: {flow_count}")

def monitor_flows(switch_name):
    """Monitor the number of flows in the switch"""
    while True:
        flows = os.popen(f'ovs-ofctl dump-flows {switch_name} | wc -l').read()
        print(f"Current flow count: {int(flows)-1}")  # -1 to account for header
        time.sleep(1)

def create_network():
    topo = TestTopo()
    net = Mininet(
        topo=topo,
        controller=RemoteController('c0', ip='127.0.0.1', port=6653),
        switch=OVSKernelSwitch
    )
    return net

if __name__ == '__main__':
    setLogLevel('info')
    net = create_network()
    net.start()
    
    # Start flow monitoring in a separate thread
    monitor_thread = threading.Thread(
        target=monitor_flows,
        args=('s1',),
        daemon=True
    )
    monitor_thread.start()
    
    # Generate all possible flows
    generate_all_flows(net)
    
    print("\nFlow generation complete. You can now use the CLI to generate additional traffic.")
    print("Example commands:")
    print("  h1 ping h2")
    print("  h1 iperf -s &  # Start server")
    print("  h2 iperf -c h1  # Start client")
    
    CLI(net)
    net.stop() 