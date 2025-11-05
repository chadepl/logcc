""" 
Runs the pivot and cn-pivot methods with and without acceleration for a series of scenarios with synthetic data.
We focus on three parameters: the size of the grid/raster, the number of clusters (adjacent and not adjacent) and the number of ensemble members.

We pick a tuple of parameters and run the method if it has not been run in the results directory.
We run SEEDS repeats per tuple. 
We design the script in such a way that can always be restarted if it is stopped.
"""
# Standard library
from time import time
from pathlib import Path
import sys
from itertools import product
import pickle

# Third-party
import numpy as np

# Local modules
from logcc_exps.config import DATA_DIR

from logcc_exps.lib.data.grids import (
    correlation_grid_division,
    correlation_grid_aggregation
)
from logcc_exps.lib.clustering.pivot import correlation_clustering as pivot
from logcc_exps.lib.clustering.pfaffelmoser2012 import (
    correlation_clustering as pfaffelmoser2012
)
from logcc_exps.lib.clustering.logcc import (
    local_step,
    global_step_pfaffelmoser2012,
    global_step_pivot
)


def timed_run(ensemble, method, accelerate_flag, rho, local_rho, corr_fn, seed):
    
    if method == "pivot":
        if accelerate_flag:
            t_start = time()
            lcent, lclust = local_step(ensemble, rho=local_rho, corr_fn=corr_fn, seed=seed)
            gcent, gclust = global_step_pivot(ensemble, centroids=lcent, clusters=lclust, rho=rho, corr_fn=corr_fn, seed=seed)
            t_end = time()
        else:
            t_start = time()
            gcent, gclust = pivot(ensemble, rho=rho, corr_fn=corr_fn, seed=seed)
            t_end = time()
        
    elif method == "pfaffelmoser2012":
        if accelerate_flag:
            t_start = time()
            lcent, lclust = local_step(ensemble, rho=local_rho, corr_fn=corr_fn, seed=seed)
            gcent, gclust = global_step_pfaffelmoser2012(ensemble, centroids=lcent, clusters=lclust, rho=rho, corr_fn=corr_fn, seed=seed)
            t_end = time()
        else:
            t_start = time()
            pfaffelmoser2012(ensemble, rho=rho, corr_fn=corr_fn, seed=seed)
            t_end = time()
    
    t_total = t_end - t_start
    return t_total


if __name__ == "__main__":    

    rng = np.random.default_rng(seed=42)

    data_dir = DATA_DIR / "scaling_grid"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    # Fixed parameters
    NUM_TRIALS = 10 # number of trials per tuple
    N_X_POINTS = 10
    LOCAL_RHO = 0.99
    RHO = 0.7
    CORR_FN = lambda ei, ej: np.corrcoef(ei, ej)[0, 1]
    MAX_SQRT_M = dict(pivot=256, pfaffelmoser2012=88)  # budget on number of locations 

    # Parameters for grid
    NUMBER_MEMBERS = [30, ]    
    SEEDS = [42, ] + list(rng.choice(np.arange(1000), NUM_TRIALS - 1, replace=False)) # number of trials per tuple
    METHODS = [
        "pivot",
        "pfaffelmoser2012",
        ]
    ACCELERATE_FLAGS = [
        True,
        False
        ]
    PROPORTION_CLUSTERS = [
        # 1/2, # un-comment for a more extensive run
        # 3/5, # un-comment for a more extensive run
        3/4, 
        1
        ] # controls the amount of clusters with disconnected components 
    X_AXIS_IDS = np.arange(N_X_POINTS).tolist() # ids of points we use to get the right method-dependent size 

    # Simulation of increasing field size
    if False:                                        
        REG_SIZE = [
            4, 
            8
            ]        
                
        exp_name = "increasing_field_size"
        exp_path = data_dir.joinpath(exp_name)

        if not exp_path.exists():
            exp_path.mkdir(parents=True)

        iterables = [
            METHODS,
            ACCELERATE_FLAGS,
            SEEDS, # [:3], [3:6], [6:]
            NUMBER_MEMBERS,
            X_AXIS_IDS,
            REG_SIZE,
            PROPORTION_CLUSTERS
        ]
        
        iterables = list(product(*iterables))

        for i, t in enumerate(iterables):

            method, accelerate_flag, seed, n, x_axis_id, reg_size, clust_prop = t
            num_reg = np.linspace(2, MAX_SQRT_M[method]/reg_size, N_X_POINTS, dtype=int)[x_axis_id]
            sqrt_m = num_reg * reg_size
            m = sqrt_m ** 2
            num_clust = int(num_reg * num_reg * clust_prop)

            fn_base = [str(a) for a in (method, accelerate_flag, seed, n, sqrt_m, num_reg, reg_size, clust_prop)]
            fn = f"{'-'.join(fn_base)}.pkl"
            run_path = exp_path.joinpath(fn)

            status = run_path.exists()

            print(f"{i}/{len(iterables)}: {fn} - {status}")

            if not status: 
                print(" - computing ...")                
                ensemble = correlation_grid_aggregation(n, reg_size, reg_size, num_reg, num_reg, num_clust, False, seed)

                t_total = timed_run(ensemble, method, accelerate_flag, RHO, LOCAL_RHO, CORR_FN, seed)
                print(f" - done in {t_total} seconds")

                record = dict(
                    exp=exp_name,
                    method=method,
                    accelerate_flag=accelerate_flag,
                    seed=seed,
                    n=n,
                    m=m,
                    sqrt_m=sqrt_m,
                    num_reg=num_reg,
                    reg_size=reg_size,
                    num_clust=num_clust,
                    clust_prop=clust_prop,
                    time_secs=t_total)
                
                with open(run_path, "wb") as f:
                    pickle.dump(record, f)


    # Simulation of increasing correlation threshold (which affects the region sizes)
    if True:        

        exp_name = "increasing_corr_thresh"
        exp_path = data_dir.joinpath(exp_name)

        if not exp_path.exists():
            exp_path.mkdir(parents=True)

        iterables = [
            METHODS,
            ACCELERATE_FLAGS,
            SEEDS, # [:3], [3:6], [6:]
            NUMBER_MEMBERS,
            X_AXIS_IDS,
            PROPORTION_CLUSTERS
        ]
        
        iterables = list(product(*iterables))

        for i, t in enumerate(iterables):

            method, accelerate_flag, seed, n, x_axis_id, clust_prop = t
            num_reg = np.linspace(2, int(MAX_SQRT_M[method]/4), N_X_POINTS, dtype=int)[x_axis_id]         
            sqrt_m = MAX_SQRT_M[method]
            m = sqrt_m ** 2
            num_cells = num_reg * num_reg
            reg_size = int(m / num_cells)
            num_clust = int(num_cells * clust_prop)

            fn_base = [str(a) for a in (method, accelerate_flag, seed, n, sqrt_m, num_reg, reg_size, clust_prop)]
            fn = f"{'-'.join(fn_base)}.pkl"
            run_path = exp_path.joinpath(fn)

            status = run_path.exists()

            print(f"{i}/{len(iterables)}: {fn} - {status}")

            if not status: 
                print(" - computing ...")
                ensemble = correlation_grid_division(n, sqrt_m, sqrt_m, num_reg, num_reg, num_clust, False, False, seed)

                t_total = timed_run(ensemble, method, accelerate_flag, RHO, LOCAL_RHO, CORR_FN, seed)
                print(f" - done in {t_total} seconds")

                record = dict(
                    exp=exp_name,
                    method=method,
                    accelerate_flag=accelerate_flag,
                    seed=seed,
                    n=n,
                    m=m,
                    sqrt_m=sqrt_m,
                    num_reg=num_reg,
                    reg_size=reg_size,
                    num_clust=num_clust,
                    clust_prop=clust_prop,
                    time_secs=t_total)
                
                with open(run_path, "wb") as f:
                    pickle.dump(record, f)

    