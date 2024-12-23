from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import TCLink
import subprocess
import time
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import threading
from traffic_generator import TrafficGenerator

NUM_HOSTS = 20
FLOW_TABLE_LIMIT = 100  # Maximum number of flows in the table
TEST_DURATION = 180     # 3 minutes per pattern
TRAFFIC_RATE = 50      # packets per second

class TestTopo(Topo):
    def build(self):
        # Add one switch
        s1 = self.addSwitch('s1')
        
        # Add 30 hosts
        for i in range(NUM_HOSTS):
            host = self.addHost(f'h{i+1}')
            self.addLink(host, s1, cls=TCLink, bw=100)

class ControllerTest:
    def __init__(self):
        self.results_dir = "test_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.traffic_gen = TrafficGenerator(num_hosts=NUM_HOSTS)  # Match number of hosts in topology
        
    def start_controller(self, controller_type):
        """Start the specified controller"""
        if controller_type == "DQN":
            cmd = ["ryu-manager", "RL_controller.py"]
        elif controller_type == "FIFO":
            cmd = ["ryu-manager", "fifo_controller.py"]
        else:  # LRU
            cmd = ["ryu-manager", "lru_controller.py"]
            
        print(f"Starting {controller_type} controller...")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def run_traffic_test(self, test_type="uniform"):
        """Run different types of traffic patterns"""
        print(f"\nRunning {test_type} traffic pattern...")
        
        self.traffic_gen.send_traffic(
            pattern=test_type,
            duration=TEST_DURATION,  # Updated duration
            rate=TRAFFIC_RATE       # Updated rate
        )

    def collect_metrics(self, duration=60):
        """Collect flow table and performance metrics"""
        metrics = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Get flow table size and details
            flow_output = subprocess.check_output(
                "sudo ovs-ofctl dump-flows s1", 
                shell=True
            ).decode()
            
            flow_count = len([line for line in flow_output.split('\n') if 'actions=' in line])
            
            # Calculate table utilization
            table_utilization = (flow_count / FLOW_TABLE_LIMIT) * 100
            
            metrics.append({
                'timestamp': time.time() - start_time,
                'flow_count': flow_count,
                'table_utilization': table_utilization,
                'bytes_transmitted': self.get_bytes_transmitted()
            })
            
            time.sleep(1)
            
        return metrics

    def get_bytes_transmitted(self):
        """Helper method to get total bytes transmitted"""
        port_stats = subprocess.check_output(
            "sudo ovs-ofctl dump-ports s1", 
            shell=True
        ).decode()
        
        return sum(int(line.split("tx bytes=")[1].split()[0]) 
                  for line in port_stats.split('\n') 
                  if "tx bytes=" in line)

    def test_controller(self, controller_type):
        """Test a single controller with all traffic patterns"""
        print(f"\nTesting {controller_type} controller...")
        
        # Start controller
        controller = self.start_controller(controller_type)
        time.sleep(5)
        
        # Create and start network
        topo = TestTopo()
        net = Mininet(topo=topo, controller=RemoteController('c0', ip='127.0.0.1'),
                     switch=OVSKernelSwitch, link=TCLink)
        net.start()
        time.sleep(5)
        
        # Start metrics collection in a separate thread
        metrics = []
        stop_collection = threading.Event()
        
        def collect_metrics_thread():
            while not stop_collection.is_set():
                metrics.extend(self.collect_metrics(duration=1))
        
        collector = threading.Thread(target=collect_metrics_thread)
        collector.start()
        
        # Run different traffic patterns
        for pattern in ["uniform", "bursty", "periodic"]:
            self.run_traffic_test(pattern)
        
        # Stop metrics collection and cleanup
        stop_collection.set()
        collector.join()
        
        # Save results
        self.save_metrics(metrics, controller_type)
        
        # Cleanup
        net.stop()
        controller.terminate()
        subprocess.run(["sudo", "mn", "-c"])
        time.sleep(5)
        
        return metrics

    def save_metrics(self, metrics, controller_type):
        """Save metrics to CSV file"""
        filename = f"{self.results_dir}/{controller_type}_metrics_{self.timestamp}.csv"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    def plot_results(self, all_metrics):
        """Generate comparison plots"""
        plt.figure(figsize=(15, 10))
        
        # Plot flow table utilization
        plt.subplot(2, 1, 1)
        for controller, metrics in all_metrics.items():
            times = [m['timestamp'] for m in metrics]
            flows = [m['flow_count'] for m in metrics]
            plt.plot(times, flows, label=controller)
        
        plt.title('Flow Table Utilization')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Number of Flows')
        plt.axhline(y=100, color='r', linestyle='--', label='Table Limit')
        plt.legend()
        plt.grid(True)
        
        # Plot bandwidth utilization
        plt.subplot(2, 1, 2)
        for controller, metrics in all_metrics.items():
            times = [m['timestamp'] for m in metrics]
            bandwidth = [m['bytes_transmitted'] / 1000000 for m in metrics]  # Convert to MB
            plt.plot(times, bandwidth, label=controller)
        
        plt.title('Bandwidth Utilization')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Total Bytes Transmitted (MB)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/comparison_{self.timestamp}.png")
        plt.close()

def main():
    setLogLevel('info')
    tester = ControllerTest()
    
    # Test each controller
    # controllers = ["FIFO", "LRU", "DQN"]
    controllers = ["FIFO", "LRU"]
    all_metrics = {}
    
    for controller in controllers:
        metrics = tester.test_controller(controller)
        all_metrics[controller] = metrics
    
    # Generate comparison plots
    tester.plot_results(all_metrics)
    
    # Print summary statistics
    print("\nTest Results Summary:")
    for controller, metrics in all_metrics.items():
        avg_flows = sum(m['flow_count'] for m in metrics) / len(metrics)
        max_flows = max(m['flow_count'] for m in metrics)
        total_bytes = max(m['bytes_transmitted'] for m in metrics) / 1000000  # MB
        
        print(f"\n{controller} Controller:")
        print(f"  Average flows: {avg_flows:.2f}")
        print(f"  Maximum flows: {max_flows}")
        print(f"  Total data transferred: {total_bytes:.2f} MB")

if __name__ == '__main__':
    main()