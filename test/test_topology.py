from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import sys
import time
import subprocess
import re
import random

class SingleSwitchTopo(Topo):
    def __init__(self, n=20):
        Topo.__init__(self)
        
        # Add switch
        switch = self.addSwitch('s1')
        
        # Add hosts
        hosts = []
        for h in range(n):
            host = self.addHost(f'h{h+1}', ip=f'10.0.0.{h+1}')
            self.addLink(host, switch)

def run_test(num_hosts=20, controller_type='fifo'):
    # Create topology
    topo = SingleSwitchTopo(num_hosts)
    
    # Create network with remote controller at 172.22.239.130:6633
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='172.22.239.130', port=6633)
    )
    
    # Start network
    net.start()
    hosts = [net.get(f'h{i}') for i in range(1, num_hosts + 1)]
    
    # Start iperf servers on all hosts
    info("Starting iperf servers on all hosts...\n")
    for i in range(1, num_hosts + 1):
        host = hosts[i-1]
        port = 5000 + i
        host.cmd(f'iperf -s -p {port} > iperf_server_h{i}.log 2>&1 &')

    # Allow servers to start
    time.sleep(2)

    # Read flows from generated file and execute iperf commands
    info("Executing iperf tests...\n")
    start_time = time.time()
    flow_count = 1
    stats = []
    
    with open(f'{num_hosts}_hosts_test.txt', 'r') as f:
        for line in f:
            src, dst, bandwidth, duration = line.strip().split()
            src_idx = int(src) - 1  # Convert to 0-based index
            dst_idx = int(dst) - 1  # Convert to 0-based index
            
            # Generate the specified number of flows for this pair
            src_host = hosts[src_idx]
            dst_host = hosts[dst_idx]
            bandwidth = int(bandwidth)
            duration = int(duration)
            dst_port = 5000 + dst_idx + 1
            info(f"Flow {flow_count}: h{src_idx+1} -> h{dst_idx+1}, bandwidth={bandwidth}Mbps, duration={duration}s\n")
            
            # Run iperf
            result = src_host.cmd(f'iperf -c {dst_host.IP()} -p {dst_port} -t {duration} -b {bandwidth}M')
            info(result + "\n")

            # Parse iperf results for throughput and packet loss
            throughput = 0
            packet_loss = 0
            lines = result.splitlines()
            for line in lines:
                if 'Mbits/sec' in line and 'sec' in line:
                    parts = line.split()
                    for part in parts:
                        if 'Mbits/sec' in part:
                            idx = parts.index(part) - 1
                            throughput = float(parts[idx])
                            break
                if 'lost' in line and '%' in line:
                    match = re.search(r'(\d+)/\s*\d+\s*\((\d+\.\d+)%\)', line)
                    if match:
                        packet_loss = float(match.group(2))

            # Measure latency using ping
            ping_result = src_host.cmd(f'ping -c 4 {dst_host.IP()}')
            avg_latency = 0
            for line in ping_result.splitlines():
                if 'rtt min/avg/max' in line:
                    avg_latency = float(line.split('=')[1].split('/')[1])

            # Store statistics for this flow
            flow_stats = {
                'flow': flow_count,
                'src': f'h{src_idx+1}',
                'dst': f'h{dst_idx+1}',
                'throughput': throughput,
                'packet_loss': packet_loss,
                'avg_latency': avg_latency
            }
            stats.append(flow_stats)
            
            time.sleep(0.01)  # Small delay between flows
            flow_count += 1

    # Wait for all flows to complete
    time.sleep(15)

    # Collect flow statistics from the switch
    info("Collecting flow statistics...\n")
    flow_stats = subprocess.run("sudo ovs-ofctl dump-flows s1", shell=True, capture_output=True, text=True)
    flow_count = len([line for line in flow_stats.stdout.split('\n') if 'priority=' in line])
    info(f"Number of flows in switch: {flow_count}\n")

    # Calculate flow installation statistics
    info("Calculating flow installation statistics...\n")
    total_install_time = 0
    flow_removals = 0
    cycle_count = len(stats)  # Each flow represents one installation cycle
    
    # Get flow installation time from switch statistics
    flow_stats_output = subprocess.run("sudo ovs-ofctl dump-flows s1", shell=True, capture_output=True, text=True)
    flow_lines = [line for line in flow_stats_output.stdout.split('\n') if 'duration=' in line]
    
    for line in flow_lines:
        duration_match = re.search(r'duration=(\d+\.\d+)s', line)
        if duration_match:
            total_install_time += float(duration_match.group(1))
            
    # Count removed flows by comparing installed vs current flows
    flow_removals = flow_count - len(flow_lines)
    
    info(f"Total flow removals: {flow_removals}\n")
    if cycle_count > 0:
        avg_install_time = total_install_time / cycle_count
        info(f"Total flow installation time: {total_install_time:.3f} seconds\n")
        info(f"Average flow installation time: {avg_install_time:.3f} seconds\n")

    # Print summary statistics
    info("\nSummary Statistics:\n")
    info("------------------\n")
    total_throughput = sum(stat['throughput'] for stat in stats)
    avg_latency_all = sum(stat['avg_latency'] for stat in stats) / len(stats)
    avg_packet_loss = sum(stat['packet_loss'] for stat in stats) / len(stats)
    
    # Write results to file
    with open(f'{controller_type}_results.txt', 'w') as f:
        f.write("Summary Statistics\n")
        f.write("------------------\n")
        f.write(f"Total Throughput: {total_throughput:.2f} Mbits/sec\n")
        f.write(f"Average Latency: {avg_latency_all:.2f} ms\n") 
        f.write(f"Average Packet Loss: {avg_packet_loss:.2f}%\n")
    
    # Also print to console
    info(f"Total Throughput: {total_throughput:.2f} Mbits/sec\n")
    info(f"Average Latency: {avg_latency_all:.2f} ms\n")
    info(f"Average Packet Loss: {avg_packet_loss:.2f}%\n")

    # Keep network running for manual inspection
    CLI(net)
    
    # Stop network
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    controller_type = sys.argv[1] if len(sys.argv) > 1 else 'fifo'
    num_hosts = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run_test(num_hosts, controller_type)