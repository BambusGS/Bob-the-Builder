from os.path import join
import os
import sys
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import time


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

# --- CPU
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            print(f"Converged after {i} iterations with max delta {delta:.2e}")
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }
    
    
def visualize(us, building_ids, fname='buildings.png'):
    """
    Visualizes up to 4 building temperature distributions in a 2x2 grid.
    """
    n_plots = min(len(us), 4)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten() 
    
    for i in range(4):
        ax = axes[i]
        if i < n_plots:
            im = ax.imshow(us[i][1:-1, 1:-1], cmap='inferno') 
            ax.set_title(f"Building ID: {building_ids[i]}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        
    plt.tight_layout()    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {save_path}")
    
    
# --- CUDA 2D
@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):    
    # Map grid direct to the interior: 1 to h-2, 1 to w-2 (shift +1, check h-1 and w-1)
    j_inner, i_inner = cuda.grid(2) # the swap trick cut it down to about a half per kernel call
    i = i_inner + 1
    j = j_inner + 1
    
    h, w = u.shape
    if i >= h - 1 or j >= w - 1:
        return
    
    if interior_mask[i - 1, j - 1]:
        u_new[i, j] = 0.25 * (
            u[i, j - 1] + 
            u[i, j + 1] + 
            u[i - 1, j] + 
            u[i + 1, j]
        )
    else:
        u_new[i, j] = u[i, j]

# This works nice, but looking at nsys profiler, I could see that for every call 1/3 of time is spend in kernel invocation
def jacobi_numba(u, interior_mask, max_iter):
    # Allocate GPU memory
    d_u = cuda.to_device(u)
    d_u_new = cuda.device_array_like(d_u) 
    d_u_new.copy_to_device(d_u) # Copy on-device to save about 1ms per building :(
    d_mask = cuda.to_device(interior_mask)
    
    # Configure grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(u.shape[1] / threads_per_block[0])) # perform grid swapping trick
    blocks_per_grid_y = int(np.ceil(u.shape[0] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    for i in range(max_iter):
        # Run kernel
        jacobi_kernel[blocks_per_grid, threads_per_block](d_u, d_u_new, d_mask)
        
        # Ping-pong arrays to avoid copying memory mapping
        d_u, d_u_new = d_u_new, d_u 
        
    return d_u.copy_to_host()



# --- CUDA 3D
@cuda.jit
def jacobi_kernel_3d(u, u_new, interior_mask):
    # Map grid direct to the interior: 1 to h-2, 1 to w-2 (shift +1, check h-1 and w-1)
    # Swap j (cols) and i (rows) for memory coalescing. k is the building index.
    j_inner, i_inner, k = cuda.grid(3) 
    i = i_inner + 1
    j = j_inner + 1
    
    # u shape is now (N_buildings, height, width) the thread index k maps directly to the building index
    N_b, h, w = u.shape
    
    if k >= N_b or i >= h - 1 or j >= w - 1:
        return
    
    if interior_mask[k, i - 1, j - 1]:
        u_new[k, i, j] = 0.25 * (
            u[k, i, j - 1] + 
            u[k, i, j + 1] + 
            u[k, i - 1, j] + 
            u[k, i + 1, j]
        )
    else:
        u_new[k, i, j] = u[k, i, j]
# This fixes the kernel invocation overhead by launching a single kernel for all buildings at once, making the time drop by 2/3!
def jacobi_numba_batched(all_u, all_interior_mask, max_iter):
    # Allocate GPU memory for the entire batch
    d_u = cuda.to_device(all_u)
    d_u_new = cuda.device_array_like(d_u)
    d_u_new.copy_to_device(d_u) # Copy on-device to save about 1ms per building :(
    d_mask = cuda.to_device(all_interior_mask)
    
    # Configure 3D grid and block dimensions, still 256 TPB
    threads_per_block = (16, 16, 1) 
    
    blocks_per_grid_x = int(np.ceil(all_u.shape[2] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(all_u.shape[1] / threads_per_block[1]))
    blocks_per_grid_z = int(np.ceil(all_u.shape[0] / threads_per_block[2]))
    
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    
    for i in range(max_iter):
        # Run 3D kernel once per iteration for ALL buildings
        jacobi_kernel_3d[blocks_per_grid, threads_per_block](d_u, d_u_new, d_mask)
        
        # Ping-pong arrays
        d_u, d_u_new = d_u_new, d_u 
        
    return d_u.copy_to_host()

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    building_ids_verify = ['10000', '10334', '10786', '11117'] # Use verification with buildings that the prof gave as an example
    
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
        
    if len(sys.argv) < 3:
        VERIFY = True  # Run assertion using classical approach and cuda results
    else:
        # Convert string to boolean
        VERIFY = sys.argv[2].lower() in ('true', '1', 't', 'yes', 'y')
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    
    # -------- VERIFICATION --------
    if VERIFY:
        VERIFY_RTOL = 1e-2
        VERIFY_ATOL = 1e-1
        # Find the indices of the verify buildings in the loaded list
        verify_indices = [i for i, bid in enumerate(building_ids) if bid in building_ids_verify]
        # or manually call the first 4 buildings for verification from all building_ids
        # verify_indices =  [i for i in range(min(len(building_ids), len(building_ids_verify)))]
        N_VERIFY = len(verify_indices)
        
        if N_VERIFY == 0:
            print("None of the building_ids_verify are in the current batch of building_ids.")
        else:
            print(f"Running verification using CPU classical method for {N_VERIFY} building(s)...")
            
            # --- Extract verification batch once
            verify_u0 = all_u0[verify_indices]
            verify_masks = all_interior_mask[verify_indices]
            verified_bids = [building_ids[idx] for idx in verify_indices]
            
            # --- CPU
            verify_u = np.empty((N_VERIFY, 514, 514))
            start_time = time.time()
            for i in range(N_VERIFY):
                u_cpu = jacobi(verify_u0[i], verify_masks[i], MAX_ITER, ABS_TOL)
                verify_u[i] = u_cpu
                
            print(f"CPU verification completed at {(time.time() - start_time) * 1000 / N_VERIFY:.2f} ms per building")
                
            # --- CUDA 2D
            print(f"Running CUDA kernels for {N_VERIFY} building(s)...")
            cuda_verify_u = []
            start_time = time.time()
            for i in range(N_VERIFY):
                # Calls the Numba CUDA implementation
                u_cuda = jacobi_numba(verify_u0[i], verify_masks[i], MAX_ITER)
                cuda_verify_u.append(u_cuda)
            print(f"CUDA verification completed at {(time.time() - start_time) * 1000 / N_VERIFY:.2f} ms per building")
            
            # --- CUDA 3D
            print(f"Running CUDA batched kernel for {N_VERIFY} building(s)...")
            start_time = time.time()
            cuda_verify_u_batched = jacobi_numba_batched(verify_u0, verify_masks, MAX_ITER)
            print(f"CUDA batched verification completed at {(time.time() - start_time) * 1000 / N_VERIFY:.2f} ms per building")
                
            # --- Visualization
            visualize(verify_u[:4], verified_bids[:4], fname='buildings_CPU_verify.png')
            visualize(cuda_verify_u[:4], verified_bids[:4], fname='buildings_CUDA_verify.png')
            visualize(cuda_verify_u_batched[:4], verified_bids[:4], fname='buildings_CUDA_batched_verify.png')
            
            # --- Assertions
            print("Verifying CUDA results against CPU results...")
            for i in range(N_VERIFY):
                # Using assert_allclose to account for minor floating point differences between CPU and GPU
                # Large tolerances needed, bcs cuda runs all 20k iterations (vs CPU with ABS_TOL=1e-4) runs on average 4k iterations, so not fully converged
                np.testing.assert_allclose(verify_u[i], cuda_verify_u[i], atol=VERIFY_ATOL, rtol=VERIFY_RTOL)
                np.testing.assert_allclose(verify_u[i], cuda_verify_u_batched[i], atol=VERIFY_ATOL, rtol=VERIFY_RTOL)
                
                # Calculate the maximum atol and rtol differences for reporting
                max_abs_diff = np.abs(verify_u[i] - cuda_verify_u[i]).max()
                max_rel_diff = np.abs(verify_u[i] - cuda_verify_u[i]).max() / np.abs(verify_u[i]).max()
                print(f"Building ID {verified_bids[i]}: max abs diff = {max_abs_diff:.2e}, max rel diff = {max_rel_diff:.2e}")
                max_abs_diff_batched = np.abs(verify_u[i] - cuda_verify_u_batched[i]).max()
                max_rel_diff_batched = np.abs(verify_u[i] - cuda_verify_u_batched[i]).max() / np.abs(verify_u[i]).max()
                print(f"Building ID {verified_bids[i]} (batched): max abs diff = {max_abs_diff_batched:.2e}, max rel diff = {max_rel_diff_batched:.2e}")
                
            print("Verification passed successfully!")
        
        

    # all_u = np.empty_like(all_u0)
    # for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
    #     u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    #     all_u[i] = u
        
    # start_time = time.time()
    
    # all_u = np.empty_like(all_u0)
    # for i in range(N):
    #     u_cuda = jacobi_numba(all_u0[i], all_interior_mask[i], MAX_ITER)
    #     all_u[i] = u_cuda
        
    #     print(f"Completed building {building_ids[i]} in {(time.time() - start_time) * 1000 / (i + 1):.2f} ms per building")
    # print(f"Total time for GPU processing: {(time.time() - start_time):.2f} seconds for {N} building(s)")
    
    start_time = time.time()
    
    # Launch ALL buildings at once!
    all_u = jacobi_numba_batched(all_u0, all_interior_mask, MAX_ITER)
        
    print(f"Total time for GPU processing: {(time.time() - start_time):.2f} seconds for {N} building(s)")


    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))