import os
import re
import csv

# Regex pattern to match iperf data lines
iperf_pattern = re.compile(
    r"\[\s*\d+\] local (\d+\.\d+\.\d+\.\d+) port \d+ connected with (\d+\.\d+\.\d+\.\d+) port \d+\n"
    r"\[\s*\d+\] +([\d\.]+)- *([\d\.]+) sec +([\d\.]+) ([KMG]Bytes) +([\d\.]+) Mbits/sec"
)

def convert_to_kbytes(size, unit):
    if unit == "KBytes":
        return float(size)
    elif unit == "MBytes":
        return float(size) * 1024
    elif unit == "GBytes":
        return float(size) * 1024 * 1024
    return 0

def extract_iperf_from_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        lines = f.read()
    matches = iperf_pattern.findall(lines)
    for match in matches:
        src_ip, dst_ip, start, end, transfer, unit, bandwidth = match
        duration = float(end) - float(start)
        transfer_kb = convert_to_kbytes(transfer, unit)
        data.append((src_ip, dst_ip, duration, transfer_kb, float(bandwidth)))
    return data

def process_all_models(base_dir, models, output_file='raw_data.csv'):
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model', 'filename', 'src_ip', 'dst_ip', 'duration', 'transfer_kb', 'bandwidth_mbps'])
        for model in models:
            folder = os.path.join(base_dir, model)
            for i in range(1, 21):
                filename = f"iperf_server_h{i}.log"
                filepath = os.path.join(folder, filename)
                if os.path.exists(filepath):
                    records = extract_iperf_from_file(filepath)
                    for record in records:
                        writer.writerow([model, filename, *record])
                else:
                    print(f"❗ Missing file: {filepath}")

if __name__ == "__main__":
    # Example usage
    base_dir = "./results/"  # folder contains FIFO/, LRU/, RL/
    models = ["FIFO", "LRU", "RL"]
    process_all_models(base_dir, models)
    print("Done: raw_data.csv created.")
