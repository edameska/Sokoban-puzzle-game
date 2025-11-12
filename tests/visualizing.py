import matplotlib.pyplot as plt
import numpy as np
import glob
import re

# Parse results from txt files
files = glob.glob("*.txt")
results = {}
maps = []
pattern = re.compile(r"(\w+): Time\s*=\s*([\d\.]+)s, Path length\s*=\s*(\d+), States explored\s*=\s*(\d+)")

for file in files:
    algo_name = file.replace("_results.txt", "")
    results[algo_name] = []
    with open(file, "r") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                map_name, time_val, path_len, states = match.groups()
                if map_name not in maps:
                    maps.append(map_name)
                results[algo_name].append([float(time_val), int(path_len), int(states)])

maps.sort()
algorithms = list(results.keys())

# 1. Time (line chart)
plt.figure(figsize=(10,6))
for algo in algorithms:
    y = [results[algo][k][0] if k < len(results[algo]) else np.nan for k in range(len(maps))]
    plt.plot(maps, y, marker='o', label=algo)
plt.yscale("log")  # optional if differences are big
plt.xlabel("Map")
plt.ylabel("Time (s)")
plt.title("Algorithm Runtime per Map")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("Time_per_Map.png")
plt.close()

# 2. Path length (bar chart)
plt.figure(figsize=(10,6))
width = 0.2
x = np.arange(len(maps))
for i, algo in enumerate(algorithms):
    y = [results[algo][k][1] if k < len(results[algo]) else np.nan for k in range(len(maps))]
    plt.bar(x + i*width, y, width=width, label=algo)
plt.xticks(x + width*(len(algorithms)/2 - 0.5), maps, rotation=30)
plt.ylabel("Path length")
plt.xlabel("Map")
plt.title("Path Length per Map per Algorithm")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("PathLength_per_Map.png")
plt.close()

# 3. States explored (log-scale bar chart)
plt.figure(figsize=(10,6))
for i, algo in enumerate(algorithms):
    y = [results[algo][k][2] if k < len(results[algo]) else np.nan for k in range(len(maps))]
    plt.bar(x + i*width, y, width=width, label=algo)
plt.xticks(x + width*(len(algorithms)/2 - 0.5), maps, rotation=30)
plt.ylabel("States Explored")
plt.xlabel("Map")
plt.title("States Explored per Map per Algorithm")
plt.yscale("log")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("StatesExplored_per_Map.png")
plt.close()

print("Plots saved as PNG images.")
