import re
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


# ── Timing extraction ─────────────────────────────────────────────────────────

_patterns = {
    "install": re.compile(r"Install Flow: ([\d.]+) seconds"),
    "remove": re.compile(r"Remove Flow: ([\d.]+) seconds"),
    "decision": re.compile(r"Remove Flow Decision: ([\d.]+) seconds"),
}


def extract_timings(file_path, is_rl=False):
    total_install = total_remove = 0.0
    install_count = remove_count = 0

    with open(file_path) as f:
        for line in f:
            if m := _patterns["install"].search(line):
                total_install += float(m.group(1))
                install_count += 1
            elif m := _patterns["remove"].search(line):
                total_remove += float(m.group(1))
                remove_count += 1
            elif is_rl and (m := _patterns["decision"].search(line)):
                total_remove += float(m.group(1))

    return total_install, total_remove, install_count, remove_count


def parse_results_txt(file_path):
    throughput = latency = 0.0
    with open(file_path) as f:
        for line in f:
            if m := re.search(r"Total Throughput: ([\d.]+)", line):
                throughput = float(m.group(1))
            elif m := re.search(r"Average Latency: ([\d.]+)", line):
                latency = float(m.group(1))
    return throughput, latency


def build_summary_csv(output_file=None):
    if output_file is None:
        output_file = os.path.join(RESULTS_DIR, 'timing_summary.csv')

    models = {
        "FIFO": ("fifo_timings.log", False, "FIFO/fifo_results.txt"),
        "LRU":  ("lru_timings.log",  False, "LRU/lru_results.txt"),
        "RL":   ("rl_timings.log",   True,  "RL/RL_results.txt"),
    }

    rows = []
    for model, (timing_file, is_rl, results_file) in models.items():
        timing_path  = os.path.join(RESULTS_DIR, timing_file)
        results_path = os.path.join(RESULTS_DIR, results_file)

        if not os.path.exists(timing_path):
            print(f"Warning: {timing_file} not found, skipping {model}")
            continue

        total_install, total_remove, install_count, remove_count = \
            extract_timings(timing_path, is_rl=is_rl)

        throughput, latency = (0.0, 0.0)
        if os.path.exists(results_path):
            throughput, latency = parse_results_txt(results_path)

        rows.append({
            "Model": model,
            "Install Time (s)": round(total_install, 5),
            "Remove Time (s)": round(total_remove, 5),
            "Install Count": install_count,
            "Remove Count": remove_count,
            "Throughput (Mbps)": throughput,
            "Average Latency (ms)": latency,
        })

    fieldnames = ["Model", "Install Time (s)", "Remove Time (s)",
                  "Install Count", "Remove Count", "Throughput (Mbps)", "Average Latency (ms)"]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary saved to {output_file}")
    return output_file


# ── Plotting ──────────────────────────────────────────────────────────────────

def _bar_chart(x, values, models, title, ylabel, filename, color):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, values, 0.5, color=color, edgecolor='black')
    plt.title(title, fontsize=14)
    plt.xticks(x, models)
    plt.ylabel(ylabel)
    for bar in bars:
        h = bar.get_height()
        fmt = f'{int(h)}' if ylabel == 'Count' else f'{h:.2f}'
        plt.text(bar.get_x() + bar.get_width() / 2., h, fmt, ha='center', va='bottom')
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}")


def plot_all(csv_file=None):
    if csv_file is None:
        csv_file = os.path.join(RESULTS_DIR, 'timing_summary.csv')

    data = pd.read_csv(csv_file)
    models = data['Model'].tolist()
    x = np.arange(len(models))

    _bar_chart(x, data['Install Time (s)'],   models, 'Install Time Comparison',   'Time (s)',  'install_time_comparison.png',   'skyblue')
    _bar_chart(x, data['Remove Time (s)'],    models, 'Remove Time Comparison',    'Time (s)',  'remove_time_comparison.png',    'lightcoral')
    _bar_chart(x, data['Install Count'],      models, 'Install Count Comparison',  'Count',     'install_count_comparison.png',  'lightgreen')
    _bar_chart(x, data['Remove Count'],       models, 'Remove Count Comparison',   'Count',     'remove_count_comparison.png',   'lightpink')
    _bar_chart(x, data['Throughput (Mbps)'],  models, 'Throughput Comparison',     'Mbps',      'throughput_comparison.png',     'lightblue')
    _bar_chart(x, data['Average Latency (ms)'], models, 'Average Latency Comparison', 'ms',    'latency_comparison.png',        'lightyellow')


if __name__ == "__main__":
    csv_file = build_summary_csv()
    plot_all(csv_file)
