#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# CSV header
echo "iteration,controller,packet_size,timeout,completion_time,packet_loss" > results/controller_comparison.csv

# Function to run test with specified controller
run_test() {
    local iteration=$1
    local controller=$2
    local packet_size=$3
    local timeout=$4
    
    echo "Iteration $iteration: Testing $controller controller (packet size: ${packet_size}B, timeout: ${timeout}s)..."
    
    # Start Ryu controller in background
    ryu-manager ryu_controllers/${controller}_controller.py > /dev/null 2>&1 &
    CONTROLLER_PID=$!
    
    # Wait for controller to initialize
    sleep 5
    
    # Start Mininet topology
    sudo mn --topo single,15 --mac --switch ovs,protocols=OpenFlow13 --controller remote > /dev/null 2>&1 &
    MININET_PID=$!
    
    # Wait for network to initialize
    sleep 5
    
    # Start time measurement
    start_time=$(date +%s.%N)
    
    # Run ping tests between all hosts with specified packet size and timeout
    sudo mn --test pingall,${packet_size},${timeout}
    
    # Capture completion time
    end_time=$(date +%s.%N)
    completion_time=$(echo "$end_time - $start_time" | bc)
    
    # Get packet loss from ping statistics
    packet_loss=$(sudo mn --test pingall,${packet_size},${timeout} | grep "packet loss" | awk '{print $6}')
    
    # Record results
    echo "$iteration,$controller,$packet_size,$timeout,$completion_time,$packet_loss" >> results/controller_comparison.csv
    
    # Cleanup
    sudo mn -c > /dev/null 2>&1
    kill $CONTROLLER_PID
    kill $MININET_PID
    sleep 2
}

# Test parameters
ITERATIONS=5
PACKET_SIZES=(64 512 1024 1500)  # Different packet sizes in bytes
TIMEOUTS=(1 2 5)                 # Different timeout values in seconds

# Run tests with same parameters for both controllers in each iteration
iteration=1
for size in "${PACKET_SIZES[@]}"; do
    for timeout in "${TIMEOUTS[@]}"; do
        for i in $(seq 1 $ITERATIONS); do
            # Run both controllers with same parameters
            run_test $iteration "fifo" $size $timeout
            run_test $iteration "lru" $size $timeout
            ((iteration++))
        done
    done
done

echo "Testing complete. Results stored in results/controller_comparison.csv"

# Generate statistics using Python
python3 - <<EOF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('results/controller_comparison.csv')

# Calculate statistics grouped by controller, packet size, and timeout
stats = df.groupby(['controller', 'packet_size', 'timeout']).agg({
    'completion_time': ['mean', 'std'],
    'packet_loss': ['mean', 'std']
}).round(3)

print("\nTest Results Summary:")
print("=====================")
print(stats)

# Save statistics to file
stats.to_csv('results/statistics.csv')

# Set style for better visualizations
plt.style.use('seaborn')

# Plot 1: Completion Time vs Packet Size
plt.figure(figsize=(12, 6))
sns.boxplot(x='packet_size', y='completion_time', hue='controller', data=df)
plt.title('Completion Time vs Packet Size by Controller')
plt.xlabel('Packet Size (bytes)')
plt.ylabel('Completion Time (seconds)')
plt.savefig('results/completion_time_vs_size.png', bbox_inches='tight', dpi=300)
plt.close()

# Plot 2: Packet Loss vs Timeout
plt.figure(figsize=(12, 6))
sns.boxplot(x='timeout', y='packet_loss', hue='controller', data=df)
plt.title('Packet Loss vs Timeout by Controller')
plt.xlabel('Timeout (seconds)')
plt.ylabel('Packet Loss (%)')
plt.savefig('results/packet_loss_vs_timeout.png', bbox_inches='tight', dpi=300)
plt.close()

# Plot 3: Heatmap of average completion time
pivot_completion = df.pivot_table(
    values='completion_time',
    index=['packet_size'],
    columns=['controller', 'timeout'],
    aggfunc='mean'
)

plt.figure(figsize=(15, 8))
sns.heatmap(pivot_completion, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Average Completion Time Heatmap')
plt.savefig('results/completion_time_heatmap.png', bbox_inches='tight', dpi=300)
plt.close()

# Generate summary report
with open('results/summary_report.txt', 'w') as f:
    f.write("Performance Comparison Summary\n")
    f.write("============================\n\n")
    
    # Overall averages
    f.write("Overall Averages:\n")
    overall_stats = df.groupby('controller').agg({
        'completion_time': ['mean', 'std'],
        'packet_loss': ['mean', 'std']
    }).round(3)
    f.write(f"{overall_stats}\n\n")
    
    # Best performing configurations
    f.write("Best Configurations:\n")
    best_completion = df.loc[df.groupby('controller')['completion_time'].idxmin()]
    f.write(f"Lowest Completion Times:\n{best_completion[['controller', 'packet_size', 'timeout', 'completion_time']]}\n\n")
    
    best_loss = df.loc[df.groupby('controller')['packet_loss'].idxmin()]
    f.write(f"Lowest Packet Loss:\n{best_loss[['controller', 'packet_size', 'timeout', 'packet_loss']]}\n")
EOF
