# Standard library
from pathlib import Path
import pickle

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# Local modules
from logcc_exps.config import DATA_DIR, cat_dataset_map
from logcc_exps.lib.clustering.utils import get_clustering_img


if __name__ == "__main__":    
    
    dataset = list(cat_dataset_map.keys())[1]
    RHO = [0.3, 0.5, 0.7, 0.9][3]
    SEED = [768, 201, 42, 650, 696, 431, 436, 85, 856, 88][0]

    # Seeds per dataset:
    # - cvp meteo = 8
    # - Parotid Parotid_R = 0
    # - Parotid Brainstem = 0 (for now)

    dataset_path = DATA_DIR / f"rd_grid/{dataset}"
    print(dataset_path.exists())
    
    ensemble = np.load(dataset_path.joinpath("ensemble.npy"))
    N, ROWS, COLS = ensemble.shape

    # 1. Some samples of the ensemble (viridis color)
    for i in range(10):
        fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
        ax.imshow(ensemble[i], cmap="viridis")
        ax.set_axis_off()
        fig.savefig(f"/Users/chadepl/Downloads/{dataset_path.stem}-ensemble_member-{i}.png", dpi=300, bbox_inches='tight', pad_inches=0.0)

    # 2. The local steo results
    # 3. The global step results

    local_clusters_imgs = []
    global_clusters_imgs = []

    for d in dataset_path.glob("*.pkl"):
        with open(d, "rb") as f:
            record = pickle.load(f)

            if record["seed"] == SEED and record["method"] == "pivot" and record["accelerate_flag"]:
                record_name = f"{record['dataset']}-{record['method']}-{record['accelerate_flag']}-rho{record['rho']}-rho{record['seed']}"
                print(record_name)

                local_clusters_imgs.append(get_clustering_img(ensemble, record["local_centroids"], record["local_clusters"]))
                global_clusters_imgs.append(get_clustering_img(ensemble, record["global_centroids"], record["global_clusters"]))

                fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
                ax.imshow(local_clusters_imgs[-1], cmap="rainbow")
                ax.set_axis_off()
                fig.savefig(f"/Users/chadepl/Downloads/local-{record_name}.png", dpi=300)

                fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
                ax.imshow(global_clusters_imgs[-1], cmap="rainbow")
                ax.set_axis_off()
                fig.savefig(f"/Users/chadepl/Downloads/global-{record_name}.png", dpi=300)
