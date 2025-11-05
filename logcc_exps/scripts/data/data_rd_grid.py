""" 
Runs the pivot and cn-pivot methods with and without acceleration for three real datasets: meteo, han_brainstem, han_parotidr.
It uses different correlation thresholds and stores the clusterings and times.
"""
# Standard library
from time import time
from pathlib import Path
from itertools import product
import pickle

# Third-party
import numpy as np

# Local modules
from logcc_exps.config import DATA_DIR

from logcc_exps.lib.data.han_data import (
    hptc_brainstem,
    hptc_right_parotid,
    get_han_ensemble,
)
from logcc_exps.lib.data.meteo_data import (
    get_cvp_meteo_data,
)

from logcc_exps.lib.clustering.pivot import (
    correlation_clustering as pivot,
)
from logcc_exps.lib.clustering.pfaffelmoser2012 import (
    correlation_clustering as pfaffelmoser2012,
)
from logcc_exps.lib.clustering.logcc import (
    local_step,
    global_step_pivot,
    global_step_pfaffelmoser2012,
)


# ana_struct = [BrainStem, Parotid_R]
# range_num_slices -> added above and below the middle slice (then adds x2 the number of slices)
def hptc_3d(scale_factor=1, iso_value=0.8, only_ensemble=False, ana_struct="BrainStem", range_num_slices=5):
    img, gt, ensemble_masks = get_han_ensemble(scale_factor=scale_factor, structure_name=ana_struct, iso_value=iso_value, slice_num=None)
    ensemble_masks = np.array(ensemble_masks)
    num_slices = ensemble_masks.shape[1]
    mid_slice = num_slices // 2    
    if range_num_slices is not None:
        img, gt, ensemble_masks = img[mid_slice-range_num_slices:mid_slice+range_num_slices], gt[mid_slice-range_num_slices:mid_slice+range_num_slices], ensemble_masks[:, mid_slice-range_num_slices:mid_slice+range_num_slices]
    if only_ensemble:
        return ensemble_masks
    return img, gt, ensemble_masks


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)    

    data_dir = DATA_DIR / "rd_grid" # NOTE: if running 3d datasets (by commenting the rest): change path to rd_grid_3d, comment "pfaffelmoser 2012" and only use rho=0.7 and 0.9 (comment the rest).
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    NUM_TRIALS = 10
    LOCAL_CORR = 0.99

    # grid params
    SEEDS = [42, ] + list(rng.choice(np.arange(1000), NUM_TRIALS - 1, replace=False))
    DATASETS = [
            ("cvp_meteo", get_cvp_meteo_data, dict(scale_factor=1)),
            ("hptc_right_parotid", hptc_right_parotid, dict(scale_factor=1, iso_value=None, only_ensemble=True)),
            ("hptc_brainstem", hptc_brainstem, dict(scale_factor=1, iso_value=None, only_ensemble=True)),
            # ("hptc_right_parotid_3d", hptc_3d, dict(ana_struct="Parotid_R", only_ensemble=True, iso_value=None, range_num_slices=None)),
            # ("hptc_brainstem_3d", hptc_3d, dict(ana_struct="BrainStem", only_ensemble=True, iso_value=None, range_num_slices=None)),
        ] 
    
    METHODS = [
        "pivot", 
        "pfaffelmoser2012",
        ]
    ACCELERATE_FLAGS = [
        True, 
        False
        ]
    
    CORR_FN = lambda ei, ej: np.corrcoef(ei, ej)[0, 1]
    RHOS = [
        0.3, 
        0.4,
        0.5, 
        0.6,
        0.7,
        0.8, 
        0.9
        ]

    # save ensemble we ran the experiment on for posterior evaluation

    for dataset_name, dataset_fn, dataset_kwargs in DATASETS:
        print(f"Starting dataset {dataset_name}")        

        dataset_path = data_dir.joinpath(f"{dataset_name}")
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True)

        ensemble_path = dataset_path.joinpath(f"ensemble.npy")
        if ensemble_path.exists():
            ensemble = np.load(ensemble_path)
        else:
            ensemble = dataset_fn(**dataset_kwargs)
            
            np.save(ensemble_path, ensemble)
            print("Finishing saving ensemble")
        print(ensemble.shape)

        if len(ensemble.shape) == 3:
            N, ROWS, COLS = ensemble.shape
            M = ROWS * COLS        
        else:
            N, SLICES, ROWS, COLS = ensemble.shape
            M = SLICES * ROWS * COLS

        # run experiment

        iterables = [
                SEEDS,
                METHODS,
                ACCELERATE_FLAGS,
                RHOS,
            ]
            
        iterables = list(product(*iterables))

        for i, t in enumerate(iterables[::-1]):
            
            seed, method, accelerate_flag, rho = t

            fn_base = [str(a) for a in (method, accelerate_flag, rho, seed)]
            fn = f"{'-'.join(fn_base)}.pkl"
            run_path = dataset_path.joinpath(fn)

            status = run_path.exists()

            print(f"{i}/{len(iterables)} [{dataset_name}]: {fn} - {status}")

            if not status: 

                if accelerate_flag:
                    # local step
                    t_start = time()
                    local_centroids, local_clusters = local_step(ensemble,
                                                                rho=LOCAL_CORR,
                                                                corr_fn=CORR_FN,
                                                                neighborhood_mode=2,
                                                                seed=seed)
                    t_end = time()
                    t_local = t_end - t_start
                    print("Finished local")

                    # global step
                    t_start = time()
                    if method == "pivot":
                        global_centroids, global_clusters = global_step_pivot(ensemble,
                                                                        centroids=local_centroids,
                                                                        clusters=local_clusters,
                                                                        rho=rho,
                                                                        corr_fn=CORR_FN,
                                                                        visiting_order="centroids",
                                                                        seed=seed)
                    elif method == "pfaffelmoser2012":
                        global_centroids, global_clusters = global_step_pfaffelmoser2012(ensemble,
                                                                                        centroids=local_centroids,
                                                                                        clusters=local_clusters,
                                                                                        rho=rho,
                                                                                        corr_fn=CORR_FN,
                                                                                        visiting_order="centroids",
                                                                                        seed=seed)
                    else:
                        raise Exception("Unrecognized method")
                    t_end = time()
                    t_global = t_end - t_start
                    print("Finished global")

                else:
                    local_centroids = []
                    local_clusters = {}
                    t_local = 0

                    # global step
                    t_start = time()
                    if method == "pivot":
                        global_centroids, global_clusters = pivot(ensemble,
                                                                rho=rho,
                                                                corr_fn=CORR_FN,
                                                                seed=seed)
                    elif method == "pfaffelmoser2012":
                        global_centroids, global_clusters = pfaffelmoser2012(ensemble,
                                                                            rho=rho,
                                                                            corr_fn=CORR_FN,
                                                                            seed=seed)
                    else:
                        raise Exception("Unrecognized method")
                    t_end = time()
                    t_global = t_end - t_start            
                
                t_total = t_local + t_global

                record = {
                    "dataset":dataset_name,
                    "seed":seed,
                    "method":method,
                    "accelerate_flag":accelerate_flag, 
                    "rho":rho,
                    "local_rho": LOCAL_CORR if accelerate_flag else "NA",
                    "local_centroids":local_centroids,
                    "local_clusters":local_clusters,
                    "global_centroids":global_centroids,
                    "global_clusters":global_clusters,
                    "t_local":t_local,
                    "t_global":t_global,
                    "t_total":t_total,
                    "fn":str(run_path)
                }

                with open(run_path, "wb") as f:
                    pickle.dump(record, f)