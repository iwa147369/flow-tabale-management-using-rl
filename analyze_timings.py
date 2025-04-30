import re
import os
import csv

# Regex patterns
patterns = {
    "install": re.compile(r"Install Flow: ([\d.]+) seconds"),
    "remove": re.compile(r"Remove Flow: ([\d.]+) seconds"),
    "decision": re.compile(r"Remove Flow Decision: ([\d.]+) seconds"),
}

def extract_timings(file_path, is_rl=False):
    total_install = 0.0
    total_remove = 0.0
    install_count = 0
    remove_count = 0

    with open(file_path, "r") as f:
        for line in f:
            if m := patterns["install"].search(line):
                total_install += float(m.group(1))
                install_count += 1
            elif m := patterns["remove"].search(line):
                total_remove += float(m.group(1))
                remove_count += 1
            elif is_rl and (m := patterns["decision"].search(line)):
                total_remove += float(m.group(1))  # Add to RL's remove total time only

    return total_install, total_remove, install_count, remove_count

def analyze_and_save_to_csv():
    models = {
        "FIFO": ("fifo_timings.log", False),
        "LRU": ("lru_timings.log", False),
        "RL": ("rl_timings.log", True)
    }

    summary_rows = []

    for model, (filename, is_rl) in models.items():
        if not os.path.exists(filename):
            print(f"⚠️  File not found: {filename}")
            continue
        total_install, total_remove, install_count, remove_count = extract_timings(filename, is_rl=is_rl)
        summary_rows.append({
            "Model": model,
            "Install Time (s)": round(total_install, 5),
            "Remove Time (s)": round(total_remove, 5),
            "Install Count": install_count,
            "Remove Count": remove_count
        })

    # Save to CSV
    with open("timing_summary.csv", "w", newline="") as csvfile:
        fieldnames = ["Model", "Install Time (s)", "Remove Time (s)", "Install Count", "Remove Count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(summary_rows)

    print("✅ Summary saved to timing_summary.csv")

if __name__ == "__main__":
    analyze_and_save_to_csv()
