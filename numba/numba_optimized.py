import sys
import time
from itertools import repeat
from multiprocessing.pool import Pool
from os.path import join

import numpy as np

from numba import jit, prange


def progress_tracker(
    iterable, total=None, prefix="Progress:", suffix="Complete", length=50
):
    """
    A simple progress tracker with ETA that can be wrapped around any iterable.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = 0

    start_time = time.time()

    def print_bar(iteration, total):
        if total == 0:
            return
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = "█" * filled_length + "-" * (length - filled_length)

        elapsed = time.time() - start_time
        if iteration > 0:
            eta = (elapsed / iteration) * (total - iteration)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        else:
            eta_str = "--:--:--"

        sys.stderr.write(f"\r{prefix} |{bar}| {percent}% {suffix} | ETA: {eta_str}")
        sys.stderr.flush()

    print_bar(0, total)
    for i, item in enumerate(iterable):
        yield item
        print_bar(i + 1, total)
    sys.stderr.write("\n")
    sys.stderr.flush()


def load_data(load_dir, bid):
    SIZE = 512
    # Ensure arrays are C-contiguous for better performance
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return np.ascontiguousarray(u), np.ascontiguousarray(interior_mask)


@jit(nopython=True, fastmath=True, cache=True)
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    """
    Optimized Jacobi iteration. 
    Parallelism is handled by the multiprocessing Pool at the building level
    to avoid oversubscription, while fastmath enables SIMD vectorization.
    """
    rows, cols = u.shape
    u_curr = u.copy()
    u_next = u.copy()

    for i in range(max_iter):
        delta = 0.0
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if interior_mask[r - 1, c - 1]:
                    # Using local variables to help the compiler vectorize
                    val = 0.25 * (
                        u_curr[r, c - 1]
                        + u_curr[r, c + 1]
                        + u_curr[r - 1, c]
                        + u_curr[r + 1, c]
                    )
                    u_next[r, c] = val
                    
                    diff = abs(val - u_curr[r, c])
                    if diff > delta:
                        delta = diff

        # Swap buffers
        u_curr, u_next = u_next, u_curr

        if delta < atol:
            break
            
    return u_curr


@jit(nopython=True, fastmath=True, cache=True)
def summary_stats_jit(u, interior_mask):
    """
    Single-pass calculation of stats using Welford's algorithm for stability.
    """
    u_interior = u[1:-1, 1:-1]
    
    count = 0
    mean = 0.0
    m2 = 0.0
    above_18 = 0
    below_15 = 0
    
    rows, cols = u_interior.shape
    for r in range(rows):
        for c in range(cols):
            if interior_mask[r, c]:
                val = u_interior[r, c]
                count += 1
                
                # Welford's algorithm for mean and variance
                delta_val = val - mean
                mean += delta_val / count
                delta2 = val - mean
                m2 += delta_val * delta2
                
                if val > 18:
                    above_18 += 1
                elif val < 15:
                    below_15 += 1
                    
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    std_temp = np.sqrt(m2 / count)
    pct_above_18 = (above_18 / count) * 100.0
    pct_below_15 = (below_15 / count) * 100.0
    
    return mean, std_temp, pct_above_18, pct_below_15


def summary_stats(u, interior_mask):
    """
    Wrapper for the JITted summary statistics.
    """
    mean, std, above_18, below_15 = summary_stats_jit(u, interior_mask)
    return {
        "mean_temp": mean,
        "std_temp": std,
        "pct_above_18": above_18,
        "pct_below_15": below_15,
    }


def jacobi_wrapper(args):
    """Helper to unpack arguments for imap"""
    return jacobi(*args)


if __name__ == "__main__":
    LOAD_DIR = "./data/"
    try:
        with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
            building_ids = f.read().splitlines()
    except FileNotFoundError:
        building_ids = []

    if len(sys.argv) < 2:
        n = min(1, len(building_ids))
    else:
        n = int(sys.argv[1])
    building_ids = building_ids[:n]

    all_u0 = np.empty((n, 514, 514), order="C")
    all_interior_mask = np.empty((n, 512, 512), dtype="bool", order="C")

    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # Prepare arguments for parallel execution
    tasks = list(
        zip(
            all_u0,
            all_interior_mask,
            repeat(MAX_ITER),
            repeat(ABS_TOL),
        )
    )

    # We use pool.imap to enable real-time progress tracking
    all_u = []
    with Pool(processes=min(16, n)) as pool:
        # Wrap the imap iterator with our progress tracker
        for result in progress_tracker(pool.imap(jacobi_wrapper, tasks), total=n):
            all_u.append(result)

    # Print summary statistics in CSV format
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
