# Standard library
from pathlib import Path
import pickle

# Third-party
import distinctipy
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian

# Local modules
from logcc_exps.config import DATA_DIR, cat_dataset_map
from logcc_exps.lib.clustering.utils import (
    get_clustering_img,
    get_multi_comparable_colorings,
)
from logcc_exps.lib.data.han_data import get_han_ensemble


def get_selected_runs(dataset_path, method, seed):
    selected_runs_dict = {}

    for d in dataset_path.glob("*.pkl"):
        with open(d, "rb") as f:
            record = pickle.load(f)

            if record["seed"] == seed and record["method"] == method:
                dataset_name = record["dataset"]
                method_name = record["method"]
                rho = record["rho"]
                accelerate_flag = record["accelerate_flag"]
                t_total = record["t_total"]
                seed = record["seed"]
                num_clusters = len(record["global_centroids"])

                print(dataset_name, method_name, rho, accelerate_flag, t_total, num_clusters)
                record_name = f"{dataset_name}-{method_name}-{accelerate_flag}-{seed}-{rho}"

                if accelerate_flag not in selected_runs_dict:
                    selected_runs_dict[accelerate_flag] = {}
                    selected_runs_dict[accelerate_flag]["global_res"] = {}
                    selected_runs_dict[accelerate_flag]["local_res"] = {}

                if rho not in selected_runs_dict[accelerate_flag]["global_res"]:
                    selected_runs_dict[accelerate_flag]["global_res"][rho] = {}  
                                                 
                selected_runs_dict[accelerate_flag]["global_res"][rho]["record_name"] = record_name
                selected_runs_dict[accelerate_flag]["global_res"][rho]["centroids"] = record["global_centroids"]
                selected_runs_dict[accelerate_flag]["global_res"][rho]["clusters"] = record["global_clusters"]

                selected_runs_dict[accelerate_flag]["global_res"][rho]["t_local"] = record["t_local"]
                selected_runs_dict[accelerate_flag]["global_res"][rho]["t_global"] = record["t_global"]
                selected_runs_dict[accelerate_flag]["global_res"][rho]["t_total"] = record["t_total"]

                if record["accelerate_flag"]:
                    selected_runs_dict[accelerate_flag]["local_res"]["local_centroids"] = record["local_centroids"]
                    selected_runs_dict[accelerate_flag]["local_res"]["local_clusters"] = record["local_clusters"]

      

    return selected_runs_dict


if __name__ == "__main__":

    PASTEL_FACTOR = 0.3
    PLOT_SEG = True
    METHOD = ["pivot", "pfaffelmoser2012"][1]
    RHOS = [0.3, 0.5, 0.7, 0.9]
    SEED = [768, 201, 42, 650, 696, 431, 436, 85, 856, 88][0]
    datasets = list(cat_dataset_map.keys())
    dataset = datasets[2]

    structure_name = "Parotid_R" if dataset == datasets[1] else "BrainStem"
    ct, gt, ensemble_raw = get_han_ensemble(scale_factor=1, structure_name=structure_name, iso_value=None)

    gt_smooth = gt.astype(float) 
    for _ in range(2):
        gt_smooth = gaussian(gt_smooth, sigma=2)

    # Seeds per dataset:
    # - cvp meteo = 8
    # - Parotid Parotid_R = 0
    # - Parotid Brainstem = 0 (for now)

    dataset_path = DATA_DIR / f"rd_grid_v1/{dataset}"
    print(dataset_path.exists())
    
    ensemble = np.load(dataset_path.joinpath("ensemble.npy"))
    N, ROWS, COLS = ensemble.shape
    
    if True:
        ######################
        # Global clusterings #
        ######################
        print("Plotting global clustering...")    

        selected_runs = get_selected_runs(dataset_path, METHOD, SEED)

        selected_run = selected_runs[True]
        selected_run_global = selected_run["global_res"] # we want to analyze the accelerated one

        rhos = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3] # reference and the rest
        record_names = [selected_run_global[rho]["record_name"] for rho in rhos]
        centroids_arr = [selected_run_global[rho]["centroids"] for rho in rhos]
        clusters_arr = [selected_run_global[rho]["clusters"] for rho in rhos]
        clustering_img_arr = [get_clustering_img(ensemble, centroids, clusters) for centroids, clusters in zip(centroids_arr, clusters_arr)]

        new_clustering_img_arr = get_multi_comparable_colorings(clustering_img_arr[0], *clustering_img_arr[1:])
        unique_ref = np.unique(new_clustering_img_arr[0])
        colors = distinctipy.get_colors(len(unique_ref), pastel_factor=PASTEL_FACTOR)
        cm = distinctipy.get_colormap(colors)
        
        for i, rho in enumerate(rhos):
            fig, ax = plt.subplots(figsize=(5,5), layout="tight")
            ax.imshow(new_clustering_img_arr[i], cmap=cm)
            if PLOT_SEG:
                ax.contour(gt_smooth, levels=[0.5, ], colors=["black",], linewidths=[4,], linestyles=["--",])
            ax.set_axis_off()
            fig.savefig(f"/Users/chadepl/Downloads/{record_names[i]}.png", dpi=300, bbox_inches='tight', pad_inches=0.0)

    if False:
        print()
        ####################
        # Local clustering #
        ####################
        print("Plotting local clustering...")
        print("- False", np.array([v["t_total"] for _, v in selected_runs[False]["global_res"].items()]).sum())
        global_times = np.array([v["t_global"] for _, v in selected_runs[True]["global_res"].items()])
        local_time = selected_runs[False]["global_res"][0.3]["t_local"]
        print("- True", "Local:", local_time, "Global mean:", global_times.mean(), "Total: ", local_time + global_times.sum())      

        local_centroids, local_clusters = selected_run["local_res"]["local_centroids"], selected_run["local_res"]["local_clusters"]
        local_clustering_img = get_clustering_img(ensemble, local_centroids, local_clusters)

        unique_local = np.unique(local_clustering_img)
        colors = distinctipy.get_colors(len(unique_local), pastel_factor=PASTEL_FACTOR)
        cm = distinctipy.get_colormap(colors)    
        print(" - created color map")

        fig, ax = plt.subplots(figsize=(5,5), layout="tight")
        ax.imshow(local_clustering_img, cmap=cm)
        if PLOT_SEG:
            ax.contour(gt_smooth, levels=[0.5, ], colors=["black",], linewidths=[4,], linestyles=["--",])
        ax.set_axis_off()
        fig.savefig(f"/Users/chadepl/Downloads/{dataset}-local-{METHOD}-{True}.png", dpi=300, bbox_inches='tight', pad_inches=0.0)    

    if True:
        print()  
        ##################
        # General images #
        ##################
        print("Plotting general images...")

        fig, ax = plt.subplots(figsize=(5,5), layout="tight")
        ax.imshow(ct, cmap="gray", vmin=-150, vmax=250)
        if PLOT_SEG:
            ax.contour(gt_smooth, levels=[0.5, ], colors=["cyan",], linewidths=[4,], linestyles=["--",])
        ax.set_axis_off()
        fig.savefig(f"/Users/chadepl/Downloads/{structure_name}-ct.png", dpi=300, bbox_inches='tight', pad_inches=0.0)