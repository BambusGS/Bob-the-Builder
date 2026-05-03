from os.path import join
import sys
from multiprocessing import Pool
import numpy as np

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp': u_interior.mean(),
        'std_temp': u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }

def task_wrapper(args):
    """Wrapper to handle multiple arguments in imap_unordered."""
    bid, u0, interior_mask, max_iter, atol = args
    u = jacobi(u0, interior_mask, max_iter, atol)
    stats = summary_stats(u, interior_mask)
    return bid, stats

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) != 3:
        print("Usage: python dynamic_parallel.py <N_buildings> <N_processes>")
        sys.exit(1)

    N = int(sys.argv[1])
    proc = int(sys.argv[2])
    building_ids = building_ids[:N]

    # Load data (Serial phase)
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = mask

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # Prepare arguments
    task_args = [(bid, all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL) 
                 for i, bid in enumerate(building_ids)]

    # Dynamic Scheduling via Pool.imap_unordered with chunksize=1
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    results = []
    
    with Pool(processes=proc) as pool:
        # chunksize=1 ensures that workers pick up 1 task at a time (Dynamic)
        # imap_unordered yields results as soon as any worker finishes
        for bid, stats in pool.imap_unordered(task_wrapper, task_args, chunksize=1):
            results.append((bid, stats))

    # Print summary statistics in CSV format
    print('building_id, ' + ', '.join(stat_keys))
    for bid, stats in sorted(results): # Sort by bid to maintain consistent output
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
