from os.path import join
import os
import sys
import numpy as np
from time import perf_counter_ns, sleep
from tqdm import tqdm
import csv
import pandas as pd
import platform
import psutil

def get_system_info():
    info = {}

    uname = platform.uname()
    info['OS'] = uname.system + uname.release
    info['node'] = uname.node
    if uname.processor == '':
        info['CPU'] = 'none'
    else:
        info["CPU"] = uname.processor
    info['machine'] = uname.machine
    info['p_cores'] = psutil.cpu_count(logical=False)
    info['t_cores'] = psutil.cpu_count(logical=True)
    # Format into string
    result = (
        f"OS:{info['OS']},"
        f'node:{info['node']},'
        f'machine:{info['machine']},'
        f"CPU:{info['CPU']},"
        f'p_threads:{info['p_cores']},'
        f't_threads:{info['t_cores']}'
        )
    return result


def initialise():
    name_of_experiment = 'test_all_2'
    save_dir = f"output/{name_of_experiment}/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    is_present = False
    temp = np.loadtxt('script_list.csv',delimiter=',', dtype=str)
    print(temp[:,0])
    for i in temp[:,0]:
        if name_of_experiment == i:
            is_present = True

    if not is_present:
        with open("script_list.csv", 'a') as f:
            f.write(name_of_experiment + ',' + get_system_info() + '\n')
    return save_dir


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    for i in (range(max_iter)):
        jts = perf_counter_ns()
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior


        if delta < atol:
            break

    return u, i


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {'mean_temp': mean_temp, 'std_temp': std_temp,'pct_above_18': pct_above_18,'pct_below_15': pct_below_15,}



if __name__ == '__main__':

    save_dir = initialise()
    all_the_buildings = 4571

    # Load dir CHANGE TO CORRECT PATH BEFORE USING
    LOAD_DIR = '/home/david/projects/hpc/mini_project/data/modified_swiss_dwellings/'


    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    if len(sys.argv) < 2:
        N = 20
    else:
        N = int(sys.argv[1])
    

    building_ids = building_ids[:N]
    print((building_ids))
    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    bid_list = np.empty(N)
    for i, bid in (enumerate(building_ids)):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        bid_list[i] = bid
        all_interior_mask[i] = interior_mask

    
    #init save logic
    header = ["bid", 'mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15', "t_sim", "iterations"]
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    width = len(header)
    if N > 200:
        debth = 200
    else:
        debth = N
    save_data = np.empty((debth, width))
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in (enumerate(zip(all_u0,all_interior_mask))):
        ts = perf_counter_ns()
        #insert jacobi function that returns u and the number of iterations it took to reach equilibrium
        u, iterations = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u
        t_sim = perf_counter_ns() - ts
        stats = summary_stats(u, interior_mask)
        save_data[i%200, :] = [bid_list[i], *(stats[k] for k in stat_keys), t_sim, iterations]
        if i == 200:
            pd.DataFrame(save_data, columns=header).to_csv((save_dir + 'general_performance.csv'), index=False)
    


    if False:
        # Print summary statistics in CSV format
        ts = perf_counter_ns()
        print('building_id, ' + ', '.join(stat_keys)) # CSV header
        for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
            stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
        t_disp = perf_counter_ns() - ts


    pd.DataFrame(save_data, columns=header).to_csv((save_dir + 'general_performance.csv'), index=False)






