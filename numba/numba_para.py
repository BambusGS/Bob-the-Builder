import sys
from itertools import repeat
from multiprocessing.pool import Pool
from os.path import join

import numpy as np

from numba import jit


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@jit(nopython=True)
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u_curr = u.copy()
    u_next = u.copy()
    rows, cols = u_curr.shape

    for i in range(max_iter):
        delta = 0.0
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if interior_mask[r - 1, c - 1]:
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

        u_curr, u_next = u_next, u_curr

        if delta < atol:
            break
    return u_curr


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


if __name__ == "__main__":
    # Load data
    # LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    LOAD_DIR = "./data/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        n = 1
    else:
        n = int(sys.argv[1])
    building_ids = building_ids[:n]

    # Load floor plans
    all_u0 = np.empty((n, 514, 514))
    all_interior_mask = np.empty((n, 512, 512), dtype="bool")
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    with Pool(16) as pool:
        all_u = pool.starmap(
            jacobi,
            zip(
                all_u0,
                all_interior_mask,
                repeat(MAX_ITER),
                repeat(ABS_TOL),
            ),
        )

    # Print summary statistics in CSV format
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
