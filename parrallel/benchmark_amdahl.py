import subprocess
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def run_experiment(n_buildings, workers):
    """Runs orig/parralel.py and returns the wall-clock time."""
    cmd = [sys.executable, "orig/dynamic_parallel.py", str(n_buildings), str(workers)]
    start = time.perf_counter()
    # Capture output to keep benchmark logs clean
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.perf_counter()
    
    if result.returncode != 0:
        print(f"Error running experiment with {workers} workers:")
        print(result.stderr)
        return None
    
    return end - start

def main():
    # Configuration
    N_BUILDINGS = 20 # Adjust based on available data and time
    WORKERS = [1, 2, 4, 8, 12, 16]
    
    print(f"Starting Amdahl's Law Benchmark (N={N_BUILDINGS})")
    print("-" * 40)
    
    results = []
    for w in WORKERS:
        print(f"Testing {w} workers...", end=" ", flush=True)
        t = run_experiment(N_BUILDINGS, w)
        if t is not None:
            results.append(t)
            print(f"{t:.2f}s")
        else:
            print("Failed.")
    
    if not results:
        return

    times = np.array(results)
    workers = np.array(WORKERS[:len(times)])
    
    # Calculate Speedup
    t1 = times[0]
    speedup = t1 / times
    
    # Estimate Parallel Fraction (p) using Amdahl's Law
    # S(n) = 1 / ((1-p) + p/n)
    # 1/S(n) = (1-p) + p/n = 1 - p + p/n = 1 + p(1/n - 1)
    # p = (1/S(n) - 1) / (1/n - 1)
    # We'll take an average p from all points except n=1
    if len(workers) > 1:
        p_estimates = (1/speedup[1:] - 1) / (1/workers[1:] - 1)
        p_avg = np.mean(p_estimates)
    else:
        p_avg = 0
    
    theoretical_max = 1 / (1 - p_avg) if p_avg < 1 else float('inf')
    
    print("-" * 40)
    print(f"Estimated Parallel Fraction (p): {p_avg:.4f}")
    print(f"Theoretical Maximum Speedup: {theoretical_max:.2f}")
    print(f"Reasoning: The serial fraction ({1-p_avg:.4f}) is likely dominated by "
          "file I/O (loading .npy files) and process spawning overhead.")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(workers, speedup, 'o-', label='Measured Speedup')
    plt.plot(workers, workers, '--', color='gray', label='Linear Speedup (Ideal)')
    
    # Amdahl's Curve
    w_smooth = np.linspace(1, max(workers), 100)
    s_amdahl = 1 / ((1 - p_avg) + p_avg / w_smooth)
    plt.plot(w_smooth, s_amdahl, 'r:', label=f"Amdahl's Law (p={p_avg:.2f})")
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup (T1 / Tn)')
    plt.title(f"Speedup Analysis (N={N_BUILDINGS} buildings)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = "speedup_plot.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Save statistics to file
    with open("benchmark_results.txt", "w") as f:
        f.write(f"Workers,Time,Speedup\n")
        for w, t, s in zip(workers, times, speedup):
            f.write(f"{w},{t:.4f},{s:.4f}\n")
        f.write(f"\nEstimated p: {p_avg:.4f}\n")
        f.write(f"Theoretical Max Speedup: {theoretical_max:.2f}\n")

if __name__ == "__main__":
    main()
