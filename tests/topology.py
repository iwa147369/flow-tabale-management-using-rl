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
import argparse
import os
import datetime
from pathlib import Path

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

def run_test(num_hosts=20, controller_type='fifo', run_index=None, num_runs=None,
             controller_ip='127.0.0.1', controller_port=6633,
             ping_count=1, settle_time=3, traffic_file=None):
    # Resolve the traffic file before any chdir (multi-run cd's into a result dir).
    if traffic_file is None:
        traffic_file = os.path.abspath(f'{num_hosts}_hosts_test.txt')
    # Create topology
    topo = SingleSwitchTopo(num_hosts)

    # Create network with the remote Ryu controller.
    # Default 127.0.0.1 assumes Ryu runs inside the same VM as Mininet.
    # If Ryu runs on the host, pass the host bridge IP (e.g. 192.168.122.1).
    info(f"Connecting to remote controller at {controller_ip}:{controller_port}\n")
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip=controller_ip, port=controller_port)
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
    
    with open(traffic_file, 'r') as f:
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

            # Measure latency using ping (ping_count packets — fewer = faster test)
            ping_result = src_host.cmd(f'ping -c {ping_count} {dst_host.IP()}')
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

    # Wait for residual flows to settle
    time.sleep(settle_time)

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

    # Keep network running for manual inspection (only on last run or single run)
    if run_index is None or run_index == num_runs:
        CLI(net)
    
    # Stop network
    net.stop()

    return {
        "controller": controller_type,
        "run": run_index or 1,
        "total_throughput": total_throughput,
        "avg_latency": avg_latency_all,
        "avg_packet_loss": avg_packet_loss,
        "install_count": cycle_count,
        "removal_count": flow_removals,
    }


def run_multi_experiment(num_runs=1, controller_type="fifo", num_hosts=20,
                         controller_ip='127.0.0.1', controller_port=6633,
                         ping_count=1, settle_time=3):
    """Run the experiment multiple times and collect statistics."""
    results = []
    # Resolve before any chdir below — each run cd's into its own result dir.
    traffic_file = os.path.abspath(f'{num_hosts}_hosts_test.txt')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("results") / f"benchmark_{controller_type}_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, num_runs + 1):
        info(f"\n{'='*60}\n")
        info(f"Starting run {i}/{num_runs} for controller={controller_type}\n")
        info(f"{'='*60}\n")

        # Each run gets its own result file inside the run directory
        run_dir = base_dir / f"run_{i:03d}"
        run_dir.mkdir(exist_ok=True)

        # Change to run dir so result files are written there
        original_cwd = os.getcwd()
        os.chdir(run_dir)

        try:
            res = run_test(num_hosts=num_hosts, controller_type=controller_type, run_index=i, num_runs=num_runs,
                           controller_ip=controller_ip, controller_port=controller_port,
                           ping_count=ping_count, settle_time=settle_time, traffic_file=traffic_file)
            results.append(res)
        finally:
            os.chdir(original_cwd)

    # Write aggregate summary
    summary_path = base_dir / "summary.csv"
    import csv
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "controller", "total_throughput", "avg_latency", "avg_packet_loss", "install_count", "removal_count"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Print aggregate stats
    if results:
        throughputs = [r["total_throughput"] for r in results]
        latencies = [r["avg_latency"] for r in results]
        info(f"\n=== Aggregate Results ({num_runs} runs) ===\n")
        info(f"Throughput: mean={sum(throughputs)/len(throughputs):.2f}  std={ (sum((x-sum(throughputs)/len(throughputs))**2 for x in throughputs)/len(throughputs))**0.5 :.2f}\n")
        info(f"Latency:    mean={sum(latencies)/len(latencies):.3f}  std={ (sum((x-sum(latencies)/len(latencies))**2 for x in latencies)/len(latencies))**0.5 :.3f}\n")
        info(f"Full results + logs saved under: {base_dir}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mininet flow table management benchmarks")
    parser.add_argument("--controller", choices=["fifo", "lru", "rl"], default="fifo",
                        help="Which controller to test")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent runs to execute")
    parser.add_argument("--hosts", type=int, default=20,
                        help="Number of hosts in the topology")
    parser.add_argument("--controller-ip", default="127.0.0.1",
                        help="Ryu controller IP. 127.0.0.1 = Ryu in this VM; "
                             "use the host bridge IP (e.g. 192.168.122.1) if Ryu runs on the host.")
    parser.add_argument("--controller-port", type=int, default=6633,
                        help="Ryu controller OpenFlow port")
    parser.add_argument("--ping-count", type=int, default=1,
                        help="ICMP packets per flow for latency (lower = faster test; was 4)")
    parser.add_argument("--settle-time", type=int, default=3,
                        help="Seconds to wait for residual flows after the run (was 15)")
    args = parser.parse_args()

    setLogLevel("info")

    if args.runs > 1:
        run_multi_experiment(num_runs=args.runs, controller_type=args.controller, num_hosts=args.hosts,
                             controller_ip=args.controller_ip, controller_port=args.controller_port,
                             ping_count=args.ping_count, settle_time=args.settle_time)
    else:
        run_test(num_hosts=args.hosts, controller_type=args.controller,
                 controller_ip=args.controller_ip, controller_port=args.controller_port,
                 ping_count=args.ping_count, settle_time=args.settle_time)

