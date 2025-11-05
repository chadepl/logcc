# Standard library
from pathlib import Path
import pickle

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import distinctipy

# Local modules
from logcc_exps.lib.clustering.utils import get_clustering_img, get_comparable_clustering_colorings
from logcc_exps.config import DATA_DIR, cat_dataset_map, cat_method_map


if __name__ == "__main__":

    METHOD = list(cat_method_map.keys())[0]
    DATASET = list(cat_dataset_map.keys())[0]
    RHO = [0.3, 0.5, 0.7, 0.9][1]
    SEED = [42, 85, 88, 201, 431, 436, 650, 696, 768, 856][1]

    data_dir = DATA_DIR / f"rd_grid/{DATASET}"

    res_false_path = data_dir.joinpath(f"{METHOD}-False-{RHO}-{SEED}.pkl")
    res_true_path = data_dir.joinpath(f"{METHOD}-True-{RHO}-{SEED}.pkl")

    ensemble = np.load(data_dir.joinpath("ensemble.npy"))

    with open(res_false_path, "rb") as f:
        res_false = pickle.load(f)

    with open(res_true_path, "rb") as f:
        res_true = pickle.load(f)
    
    print(res_false.keys())

    cluster_img_a = get_clustering_img(ensemble, res_false["global_centroids"], res_false["global_clusters"])
    cluster_img_b = get_clustering_img(ensemble, res_true["local_centroids"], res_true["local_clusters"])


    # # complex data
    # RHO = 0.7
    # CORR_FN = lambda ei, ej: np.corrcoef(ei, ej)[0, 1]
    # # ensemble = get_cvp_meteo_data(scale_factor=1, return_masks=False)
    # # ensemble = hptc_brainstem(scale_factor=0.5, iso_value=None, only_ensemble=True)
    # ensemble = hptc_right_parotid(scale_factor=0.7, iso_value=None, only_ensemble=True)

    # print(ensemble.shape)
    
    # # lcent, lclust = local_step(ensemble, 0.99, CORR_FN)
    # t_start = time()
    # # cluster_img_a = get_clustering_img(ensemble, *pivot(ensemble, rho=RHO, corr_fn=CORR_FN, seed=42))
    # cluster_img_a = get_clustering_img(ensemble, *pfaffelmoser2012(ensemble, rho=RHO, corr_fn=CORR_FN, seed=42))
    # # cluster_img_a = get_clustering_img(ensemble, *global_step_pfaffelmoser2012_old(ensemble, lcent, lclust, rho=RHO, corr_fn=CORR_FN, seed=42))
    # print(f"Old took {time() - t_start} seconds")
    # #cluster_img_b = get_clustering_img(ensemble, *global_step_pivot(ensemble, lcent, lclust, rho=RHO, corr_fn=CORR_FN, seed=42))
    # t_start = time()
    # lcent, lclust = local_step(ensemble, 0.99, CORR_FN)
    # cluster_img_b = get_clustering_img(ensemble, *global_step_pfaffelmoser2012(ensemble, lcent, lclust, rho=RHO, corr_fn=CORR_FN, seed=42))
    # print(f"New took {time() - t_start} seconds")
    # #cluster_img_b = get_clustering_img(ensemble, lcent, lclust)
    # N, ROWS, COLS = ensemble.shape
    # # TODO: instead of running the methods load one of the files


    # colored_a, colored_b = get_comparable_clustering_colorings(ensemble, cluster_img_a, cluster_img_b) # producing weird results with cn pivot
    colored_a, colored_b = cluster_img_a, cluster_img_b

    num_clusts = np.max([np.unique(colored_a).size, np.unique(colored_b).size])
    colors = distinctipy.get_colors(num_clusts)
    cmap = distinctipy.get_colormap(colors)

    if False:
        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.suptitle(f"{DATASET}-{METHOD}-{RHO}-{SEED}")
        # previous coloring
        axs[0, 0].set_title(f"False ({res_false['t_total']} seconds)")
        axs[0, 1].set_title(f"True ({res_true['t_total']} seconds)")
        axs[0, 0].imshow(cluster_img_a.astype(int), cmap="tab10")
        axs[0, 1].imshow(cluster_img_b.astype(int), cmap="tab10")
        # improved coloring    
        axs[1, 0].imshow(colored_a, cmap="rainbow")
        axs[1, 1].imshow(colored_b, cmap="rainbow")
        # for ax in axs.flatten():
        #     #ax.set_axis_off()
        #     ax.tick_params(labelbottom=False) 
        axs[0, 0].set_ylabel("Naive approach: both clustering\n images get their own assignment")
        axs[1, 0].set_ylabel("Heuristic approach: matched\n clusters based on intersection/union")
        plt.show()

    vmin = min(colored_a.min(), colored_b.min())
    vmax = max(colored_a.max(), colored_b.max())

    # fig, ax = plt.subplots(figsize=(5,5), layout="tight")
    # ax.imshow(colored_a, cmap=cmap, vmin=vmin, vmax=vmax)
    # ax.set_axis_off()
    # # fig.savefig(f"/Users/chadepl/Downloads/{DATASET}-{METHOD}-{False}-{RHO}-{SEED}.png", dpi=300)
    # plt.show()

    fig, ax = plt.subplots(figsize=(5,5), layout="tight")
    ax.imshow(colored_b, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    # fig.savefig(f"/Users/chadepl/Downloads/{DATASET}-{METHOD}-{True}-{RHO}-{SEED}.png", dpi=300)
    plt.show()