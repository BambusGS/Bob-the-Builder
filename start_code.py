from os.path import join
import os
import sys
import numpy as np
from time import perf_counter_ns, sleep
from tqdm import tqdm
import csv
import pandas as pd



def initialise():
    name_of_experiment = 'base_case_sleep30ms_1'
    save_dir = f"output/{name_of_experiment}/"
    all_the_buildings = 4571

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    is_present = False
    temp = pd.read_csv('script_list.csv')
    for i in temp:
        if name_of_experiment == i:
            is_present = True

    if not is_present:
        with open("script_list.csv", 'a') as f:
            f.write(',' + name_of_experiment)
    return save_dir
"""
script name, GPU, GPU cach?, CPU, L1, L2, L3, OS


CPU usage per core, ram usage, current freq
add computer hardware information, CPU, GPU, total RAM usage (if possible), OS"""




def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


pbar = tqdm(total=100, desc="Jacobi Solver", leave=True)
def jacobi(u, interior_mask, max_iter, atol=1e-6, save_loc:str=None):
    u = np.copy(u)
    wd_reached = True
    jacobi_header = ['iteration', 'compute_time']
    jacobi_save_data = np.empty((max_iter, len(jacobi_header)))

    if pbar is not None:
        pbar.reset(total=max_iter)

    for i in (range(max_iter)):
        jts = perf_counter_ns()
        sleep(0.03)
        # Compute average of left, right, up and down neighbors, see eq
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        it_time = perf_counter_ns() - jts

        jacobi_save_data[i, :] = [i, it_time]
        if pbar is not None:
            pbar.set_postfix({"delta": f"{delta:.2e}"})
            pbar.update(1)

        if delta < atol:
            wd_reached = False
            if pbar is not None:
                pbar.n = max_iter # Force visual completion
                pbar.refresh()
            break

    if save_loc != None:
        pd.DataFrame(jacobi_save_data, columns=jacobi_header).to_csv(save_loc, index=False)
    return u, wd_reached, i


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {'mean_temp': mean_temp, 'std_temp': std_temp,'pct_above_18': pct_above_18,'pct_below_15': pct_below_15,}



def conv_to_s(time_ns):
    ns = time_ns%1000
    micro_s = time_ns//1000
    mili_s = micro_s//1000
    s = mili_s//1000
    micro_s = micro_s%1000
    mili_s = mili_s%1000
    time_str = f"{s}_{mili_s}_{micro_s}_{ns}"
    return time_str

"""
to do:
make a system for saving the data
get hardware info: CPU, GPU, MEM size


CLI tool for displaying progress
visualise the heat



analesys schema:
the first jacobi run each iteration time is measured and saved

on all subsequent runs:
compleate time of jacobi run, how many iterations, (remember to save the save string here too)

other data to measure:
data load time (figur out how to do this)
total time of program/projected time to finish all houses

total memory usage (optional)
total diskspace produced for data (optional)


data analesys:
plot time per iteration over iterations
plot sim time over number of iterations
compute mean sim time between scripts
compare wall time of different scripts

give each script a score based on (sim time/iterations) and wall time 


save string:
header: general info about the program
per building: time to load/prepare, time to simulate, wd triggered, number of iterations, 'mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15'

"""




if __name__ == '__main__':

    save_dir = initialise()

    # Load data
    LOAD_DIR = 'data/modified_swiss_dwellings/'
    ts = perf_counter_ns()
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    #print(len(building_ids))
    building_ids = building_ids[:N]
    
    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    bid_list = np.empty(N)
    for i, bid in (enumerate(building_ids)):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        bid_list[i] = bid
        all_interior_mask[i] = interior_mask
    t_load = perf_counter_ns() - ts



    #init save logic
    header = ["bid", 'mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15', "t_load", "t_sim", "t_disp", "wd", "iterations"]
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    width = len(header)    
    debth = N
    save_data = np.empty((debth, width))


    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    jacobi(u0, interior_mask, MAX_ITER, ABS_TOL, save_loc=(save_dir + "jacobi_performance.csv"))
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in tqdm(enumerate(zip(all_u0,all_interior_mask)), total= N):
        ts = perf_counter_ns()
        u, wd, iterations = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u
        t_sim = perf_counter_ns() - ts
        ts = perf_counter_ns()
        stats = summary_stats(u, interior_mask)
        t_disp = perf_counter_ns() - ts
        save_data[i, :] = [bid_list[i], *(stats[k] for k in stat_keys), t_load, t_sim, t_disp, int(wd), iterations]
    


    if False:
        # Print summary statistics in CSV format
        ts = perf_counter_ns()
        print('building_id, ' + ', '.join(stat_keys)) # CSV header
        for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
            stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
        t_disp = perf_counter_ns() - ts


    pd.DataFrame(save_data, columns=header).to_csv((save_dir + 'general_performance.csv'), index=False)






