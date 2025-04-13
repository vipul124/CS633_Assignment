import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Dataset identifiers and display names (simple format)
datasets = [
    ("64_64_64_3", "64_64_64_3"),
    ("64_64_96_7", "64_64_96_7")
]

process_counts = [1, 8, 16, 32, 64]
iterations = 5

def extract_times(tag):
    main_times = {p: [] for p in process_counts}
    total_times = {p: [] for p in process_counts}

    for p in process_counts:
        for i in range(1, iterations + 1):
            filename = f"output_{tag}_{p}_{i}.txt"
            if not os.path.exists(filename):
                print(f"Missing: {filename}")
                continue
            with open(filename, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"Empty: {filename}")
                    continue
                last_line = lines[-1].strip().split()
                if len(last_line) >= 3:
                    try:
                        main = float(last_line[1].replace(",", ""))
                        total = float(last_line[2].replace(",", ""))
                        main_times[p].append(main)
                        total_times[p].append(total)
                    except ValueError as e:
                        print(f"Parse error in {filename}: {e}")
                else:
                    print(f"Format error: {filename}")
    return main_times, total_times

def compute_avg(times):
    return {p: sum(times[p]) / len(times[p]) for p in process_counts if times[p]}

def plot_boxplot(data_dict, ylabel, title, filename):
    data = [data_dict[p] for p in process_counts if p != 1]
    labels = [p for p in process_counts if p != 1]
    colors = sns.color_palette("pastel")

    plt.figure(figsize=(9, 6))
    bp = plt.boxplot(data, patch_artist=True, labels=labels)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='red', linewidth=1.5)

    plt.xlabel("Number of Processes", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_speedup(avg_times, title, filename):
    base_time = avg_times[1]
    x = [p for p in process_counts if p != 1]
    speedup = [base_time / avg_times[p] for p in x]

    plt.figure(figsize=(9, 6))
    plt.plot(x, speedup, marker='o', linewidth=2, label='Speedup', color='steelblue')
    plt.plot(x, x, '--', color='gray', label='Ideal (y = x)')

    plt.xlabel("Number of Processes", fontsize=12)
    plt.ylabel("Speedup", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_efficiency(avg_times, title, filename):
    base_time = avg_times[1]
    x = [p for p in process_counts if p != 1]
    speedup = [base_time / avg_times[p] for p in x]
    efficiency = [s / p for s, p in zip(speedup, x)]

    plt.figure(figsize=(9, 6))
    plt.plot(x, efficiency, marker='s', linewidth=2, label='Efficiency', color='darkorange')
    plt.axhline(y=1.0, linestyle='--', color='gray', label='Ideal (y = 1)')
    plt.ylim(0, 1)

    plt.xlabel("Number of Processes", fontsize=12)
    plt.ylabel("Efficiency", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

# Run for both datasets
for tag, display_name in datasets:
    main_times, total_times = extract_times(tag)
    avg_main = compute_avg(main_times)
    avg_total = compute_avg(total_times)

    plot_boxplot(main_times, "Main Code Time (s)", f"{display_name} - Main Code Time vs Processes", f"box_main_{tag}.png")
    plot_boxplot(total_times, "Total Time (s)", f"{display_name} - Total Time vs Processes", f"box_total_{tag}.png")

    plot_speedup(avg_main, f"{display_name} - Speedup (Main Code Time)", f"speedup_main_{tag}.png")
    plot_efficiency(avg_main, f"{display_name} - Efficiency (Main Code Time)", f"efficiency_main_{tag}.png")

    plot_speedup(avg_total, f"{display_name} - Speedup (Total Time)", f"speedup_total_{tag}.png")
    plot_efficiency(avg_total, f"{display_name} - Efficiency (Total Time)", f"efficiency_total_{tag}.png")