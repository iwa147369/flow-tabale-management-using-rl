import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data from the CSV file
data = pd.read_csv('timing_summary.csv')

# Prepare data for plotting
models = data['Model']
install_time = data['Install Time (s)']
remove_time = data['Remove Time (s)']
install_count = data['Install Count']
remove_count = data['Remove Count']
throughput = data['Throughput (Mbps)']
latency = data['Average Latency (ms)']

# Set up positions for the bars
x = np.arange(len(models))  # Bar positions: 0, 1, 2
bar_width = 0.5  # Width of the bars

# Create individual plots for each metric

# 1. Plot Install Time
plt.figure(figsize=(8, 6))
bars = plt.bar(x, install_time, bar_width, color='skyblue', edgecolor='black')
plt.title('Install Time Comparison', fontsize=14)
plt.xticks(x, models)
plt.ylabel('Time (s)')
# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s',
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('install_time_comparison.png')
plt.close()

# 2. Plot Remove Time 
plt.figure(figsize=(8, 6))
bars = plt.bar(x, remove_time, bar_width, color='lightcoral', edgecolor='black')
plt.title('Remove Time Comparison', fontsize=14)
plt.xticks(x, models)
plt.ylabel('Time (s)')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}s',
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('remove_time_comparison.png')
plt.close()

# 3. Plot Install Count
plt.figure(figsize=(8, 6))
bars = plt.bar(x, install_count, bar_width, color='lightgreen', edgecolor='black')
plt.title('Install Count Comparison', fontsize=14)
plt.xticks(x, models)
plt.ylabel('Count')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('install_count_comparison.png')
plt.close()

# 4. Plot Remove Count
plt.figure(figsize=(8, 6))
bars = plt.bar(x, remove_count, bar_width, color='lightpink', edgecolor='black')
plt.title('Remove Count Comparison', fontsize=14)
plt.xticks(x, models)
plt.ylabel('Count')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('remove_count_comparison.png')
plt.close()

# 5. Plot Throughput
plt.figure(figsize=(8, 6))
bars = plt.bar(x, throughput, bar_width, color='lightblue', edgecolor='black')
plt.title('Throughput Comparison', fontsize=14)
plt.xticks(x, models)
plt.ylabel('Mbps')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}Mbps',
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('throughput_comparison.png')
plt.close()

# 6. Plot Average Latency
plt.figure(figsize=(8, 6))
bars = plt.bar(x, latency, bar_width, color='lightyellow', edgecolor='black')
plt.title('Average Latency Comparison', fontsize=14)
plt.xticks(x, models)
plt.ylabel('ms')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}ms',
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig('latency_comparison.png')
plt.close()