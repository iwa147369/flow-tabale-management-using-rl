import subprocess
import time
import matplotlib.pyplot as plt
from traffic_generator import TrafficGenerator

def run_controller(controller_type):
    if controller_type == "DQN":
        cmd = "ryu-manager ryu_controllers/RL_controller.py"
    elif controller_type == "FIFO":
        cmd = "ryu-manager ryu_controllers/fifo_controller.py"
    elif controller_type == "LRU":
        cmd = "ryu-manager ryu_controllers/lru_controller.py"
    return subprocess.Popen(cmd.split())

def measure_performance(controller_proc, traffic_generator, duration=300):
    metrics = {
        'packet_loss': [],
        'latency': [],
        'flow_table_utilization': []
    }
    
    # Start traffic generation
    traffic_generator.send_traffic(duration=duration)
    
    # Collect metrics (this would need to be implemented based on your specific needs)
    # You could use tools like iperf or custom packet capturing
    
    controller_proc.terminate()
    return metrics

def main():
    controllers = ["FIFO", "LRU", "DQN"]
    traffic_patterns = ["uniform", "bursty", "periodic"]
    generator = TrafficGenerator()
    
    results = {}
    
    for controller in controllers:
        results[controller] = {}
        for pattern in traffic_patterns:
            print(f"Testing {controller} with {pattern} traffic...")
            
            # Start controller
            controller_proc = run_controller(controller)
            time.sleep(5)  # Wait for controller to initialize
            
            # Run test
            metrics = measure_performance(controller_proc, generator)
            results[controller][pattern] = metrics
            
            time.sleep(2)  # Cool-down period
    
    # Plot results
    plot_comparison(results)

def plot_comparison(results):
    metrics = ['packet_loss', 'latency', 'flow_table_utilization']
    patterns = ["uniform", "bursty", "periodic"]
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(patterns))
        width = 0.25
        
        for j, controller in enumerate(['FIFO', 'LRU', 'DQN']):
            values = [results[controller][pattern][metric] for pattern in patterns]
            ax.bar(x + j*width, values, width, label=controller)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Traffic Pattern')
        ax.set_xticks(x + width)
        ax.set_xticklabels(patterns)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.close()

if __name__ == "__main__":
    main() 