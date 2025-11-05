# Standard library
from pathlib import Path
import pickle
from distinctipy import distinctipy

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Local modules
from logcc_exps.config import DATA_DIR
from logcc_exps.lib.clustering.utils import get_clustering_img, get_multi_comparable_colorings

def get_selected_runs(dataset_path, rho, seed):
    selected_runs_dict = {}

    for d in dataset_path.glob("*.pkl"):
        with open(d, "rb") as f:
            record = pickle.load(f)

            if record["rho"] == rho and record["seed"] == seed:
                dataset_name = record["dataset"]
                method_name = record["method"]
                rho = record["rho"]
                accelerate_flag = record["accelerate_flag"]
                t_total = record["t_total"]
                seed = record["seed"]
                num_clusters = len(record["global_centroids"])                    

                print(dataset_name, method_name, rho, accelerate_flag, t_total, num_clusters)
                record_name = f"{dataset_name}-{method_name}-{accelerate_flag}-{seed}-{rho}"

                if method_name not in selected_runs_dict:
                    selected_runs_dict[method_name] = {}

                if accelerate_flag not in selected_runs_dict[method_name]:
                    selected_runs_dict[method_name][accelerate_flag] = {}
                
                selected_runs_dict[method_name][accelerate_flag]["record_name"] = record_name
                selected_runs_dict[method_name][accelerate_flag]["centroids"] = record["global_centroids"]
                selected_runs_dict[method_name][accelerate_flag]["clusters"] = record["global_clusters"]

    return selected_runs_dict

if __name__ == "__main__":

    PASTEL_FACTOR = 0.3
    RHO = [0.3, 0.5, 0.7, 0.9][2]
    SEED = [768, 201, 42, 650, 696, 431, 436, 85, 856, 88][0]

    # Seeds per dataset:
    # - cvp meteo = 8
    # - Parotid Parotid_R = 0
    # - Parotid Brainstem = 0

    dataset_path = DATA_DIR / "rd_grid_v1/cvp_meteo"
    # dataset_path = Path("data/rd_grid_v1/hptc_right_parotid")
    # dataset_path = DATA_DIR / "rd_grid_v1/hptc_brainstem"
    print(dataset_path.exists())
    
    ensemble = np.load(dataset_path.joinpath("ensemble.npy"))
    N, ROWS, COLS = ensemble.shape

    selected_runs_dict = get_selected_runs(dataset_path, RHO, SEED)

    for method_name, method_items in selected_runs_dict.items():

        centroids_acc_true = method_items[True]["centroids"]
        clusters_acc_true = method_items[True]["clusters"]
        centroids_acc_false = method_items[False]["centroids"]
        clusters_acc_false = method_items[False]["clusters"]

        clustering_img_acc_true = get_clustering_img(ensemble, centroids_acc_true, clusters_acc_true)
        clustering_img_acc_false = get_clustering_img(ensemble, centroids_acc_false, clusters_acc_false)

        new_clustering_img_acc_false, new_clustering_img_acc_true = get_multi_comparable_colorings(clustering_img_acc_false, clustering_img_acc_true)

        unique_acc_false = np.unique(new_clustering_img_acc_false)
        unique_acc_true = np.unique(new_clustering_img_acc_true)

        new_clustering_imgs = [
            (new_clustering_img_acc_false, method_items[False]["record_name"]),
            (new_clustering_img_acc_true, method_items[True]["record_name"])
        ]

        colors = distinctipy.get_colors(len(unique_acc_false), pastel_factor=PASTEL_FACTOR)
        cm = distinctipy.get_colormap(colors)
        
        # Uncomment for quick inspection
        # fig, axs = plt.subplots(ncols=2)
        # axs[0].imshow(new_clustering_img_acc_false, cmap=cm)
        # axs[1].imshow(new_clustering_img_acc_true, cmap=cm)
        # plt.show()

        for cimg, file_name in new_clustering_imgs:
            fig, ax = plt.subplots(figsize=(5,5), layout="tight")
            ax.imshow(cimg, cmap=cm)
            ax.set_axis_off()
            fig.savefig(f"/Users/chadepl/Downloads/{file_name}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        