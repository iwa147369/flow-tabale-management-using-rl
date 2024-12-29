import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read data from CSV file
df = pd.read_csv('test_data.csv')

# Extract model episode numbers from model names
df['Episode'] = df['Model'].str.extract('(\d+)').astype(int)

metrics = ['Avg_Latency_ms', 'Packet_Loss_Rate', 'Throughput_Mbps', 
          'Flow_Installation_Time_ms']

# Create figure for comparison plots
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.ravel()[:4]

# Plot bar charts for each metric
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Plot bars for each host count
    episodes = df['Episode'].unique()
    x = np.arange(len(episodes))
    width = 0.25  # Width of bars
    
    for i, host_count in enumerate(df['Hosts'].unique()):
        host_data = df[df['Hosts'] == host_count]
        ax.bar(x + i*width, host_data[metric], width, 
               label=f'{host_count} hosts')
    
    # Customize plot
    ax.set_title(f'{metric} vs Training Episodes', fontsize=12, pad=20)
    ax.set_xlabel('Training Episodes', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xticks(x + width)
    ax.set_xticklabels(episodes)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
