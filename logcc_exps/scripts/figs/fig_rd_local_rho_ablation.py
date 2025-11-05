# Standard library
from pathlib import Path
import pickle

# Third-party
import distinctipy
import matplotlib.pyplot as plt
import numpy as np

# Local modules
from logcc_exps.config import DATA_DIR, cat_method_map
from logcc_exps.lib.clustering.utils import (
    get_clustering_img,
    get_multi_comparable_colorings,
)
from logcc_exps.scripts.tables.tab_rd_times import (
    load_df,
    process_df_categories,
    rename_df_categories,
)

def get_selected_runs(dataset_path, output_path, rhos, instance_seed, method_name):
    selected_runs_dict = {}

    for d in dataset_path.glob("*.pkl"):
        with open(d, "rb") as f:
            record = pickle.load(f)            

            if record["seed"] == instance_seed and record["method"] == method_name and record["rho"] in rhos:

                dataset_name = record["dataset"]
                method_name = record["method"]
                rho = record["rho"]
                local_rho = record["local_rho"]
                accelerate_flag = record["accelerate_flag"]
                t_local = record["t_local"]
                t_global = record["t_global"]
                t_total = record["t_total"]
                seed = record["seed"]
                num_clusters = len(record["global_centroids"])

                print(dataset_name, method_name, rho, accelerate_flag, t_total, num_clusters)
                record_name = f"{dataset_name}-{method_name}-{accelerate_flag}-{seed}-{rho}"       

                rho_path = output_path.joinpath(f"{rho}")
                if not rho_path.exists():
                    rho_path.mkdir()

                if rho not in selected_runs_dict:
                    selected_runs_dict[rho] = dict()

                print(dataset_name, method_name, rho, accelerate_flag, t_local, t_global, t_total)
                record_name = f"local_rho-{record['local_rho']}"
                global_centroids = record["global_centroids"]
                global_clusters = record["global_clusters"]

                selected_runs_dict[rho][local_rho] = dict(
                    record_name=record_name,
                    record_path=rho_path.joinpath(f"{record_name}.png"),
                    centroids=global_centroids,
                    clusters=global_clusters)
                
    return selected_runs_dict


if __name__ == "__main__":
    
    data_dir = DATA_DIR / "rd_grid_local_rhos"
    OUTPUT_DIR = Path("/Users/chadepl/Downloads/local_rho-ablation/")
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    PASTEL_FACTOR = 0.3
    SHOW = False
    METHOD_KEYS = list(cat_method_map.keys())
    METHOD_VALUES = list(cat_method_map.values())

    DATASETS_KEYS = ["hptc_brainstem",] # list(cat_dataset_map.keys())
    DATASETS_VALUES = ["HaN-Brainstem",] # list(cat_dataset_map.values())

    INSTANCE_SEED = 768
    INSTANCE_RHOS = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    ERROR_TYPE = ["total_edge_error_sum", "pos_edge_error_sum", "neg_edge_error_sum"][0]

    df_cfv = load_df(data_dir, which="cfv")
    df_cfv = process_df_categories(df_cfv, rhos=[0.5, 0.6, 0.7, 0.8, 0.9])
    df_cfv = rename_df_categories(df_cfv)
    df_cfv["total_edge_error_sum"] = df_cfv["pos_edge_error_sum"] + df_cfv["neg_edge_error_sum"]

    df_times = load_df(data_dir, which="times")
    df_times = process_df_categories(df_times, rhos=[0.5, 0.6, 0.7, 0.8, 0.9])
    df_times = rename_df_categories(df_times)

    print(df_times)

    # for dataset_name in DATASETS_VALUES:
    #     for method_name in METHOD_VALUES:
    #         # CFV plot
    #         df_boxplot = df_cfv.copy()
    #         df_boxplot = df_boxplot[df_boxplot["dataset"] == dataset_name]
    #         df_boxplot = df_boxplot[df_boxplot["method"] == method_name]

    #         # datset, method and accelerated flag not needed
    #         df_boxplot_means = df_boxplot.groupby(["local_rho", "rho"], observed=True)[ERROR_TYPE].mean().reset_index()

    #         fig, ax = plt.subplots(figsize=(3,3), layout="tight")
    #         sns.scatterplot(df_boxplot_means, x="local_rho", y=ERROR_TYPE, hue="rho", legend=None , ax=ax)
    #         sns.lineplot(df_boxplot, x="local_rho", y=ERROR_TYPE, hue="rho", legend=None, ax=ax)
    #         ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    #         ax.set_ylabel("Cost function value")
    #         ax.set_xlabel(r"$\rho_l$")
    #         ax.tick_params(axis='x', labelrotation=45)
            
    #         if SHOW:
    #             plt.show()
    #         else:
    #             fig.savefig(OUTPUT_DIR.joinpath(f"cfv-{dataset_name}-{method_name}-local_rho-ablation.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    #             plt.close()

    #         # Times plot
    #         df_boxplot = df_times.copy()
    #         df_boxplot = df_boxplot[df_boxplot["dataset"] == dataset_name]
    #         df_boxplot = df_boxplot[df_boxplot["method"] == method_name]
    #         print(df_boxplot)

    #         # datset, method and accelerated flag not needed
    #         df_boxplot_means = df_boxplot.groupby(["local_rho", "rho"], observed=True)["t_total"].mean().reset_index()

    #         fig, ax = plt.subplots(figsize=(3,3), layout="tight")
    #         sns.scatterplot(df_boxplot_means, x="local_rho", y="t_total", hue="rho", legend=None, ax=ax)
    #         sns.lineplot(df_boxplot, x="local_rho", y="t_total", hue="rho", legend=None, ax=ax)
    #         ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    #         ax.set_yscale('log')
    #         ax.set_ylabel("Log(Time (seconds))")
    #         ax.set_xlabel(r"$\rho_t$")
    #         ax.tick_params(axis='x', labelrotation=45)

    #         if SHOW:
    #             plt.show()
    #         else:
    #             fig.savefig(OUTPUT_DIR.joinpath(f"times-{dataset_name}-local_rho-ablation.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    #             plt.close()

    
    # Save the clusterings for a given seed

    for dataset_name in DATASETS_KEYS:
        for method_name in METHOD_KEYS:

            dataset_path = data_dir.joinpath(f"{dataset_name}")
            output_path = OUTPUT_DIR.joinpath(f"{method_name}/{dataset_name}/{INSTANCE_SEED}")
            if not output_path.exists():
                output_path.mkdir(parents=True)

            ensemble = np.load(dataset_path.joinpath("ensemble.npy"))
            N, ROWS, COLS = ensemble.shape  

            selected_runs = get_selected_runs(dataset_path, output_path, INSTANCE_RHOS, INSTANCE_SEED, method_name)
                    
            for rho_key, rho_record in selected_runs.items():
                # initialize values and add reference
                local_rhos = np.sort(list(rho_record.keys())).tolist()[::-1]
                record_names = [rho_record[local_rho]["record_name"] for local_rho in local_rhos]
                centroids_arr = [rho_record[local_rho]["centroids"] for local_rho in local_rhos]
                clusters_arr = [rho_record[local_rho]["clusters"] for local_rho in local_rhos]
                record_paths = [rho_record[local_rho]["record_path"] for local_rho in local_rhos]                
                clustering_img_arr = [get_clustering_img(ensemble, centroids, clusters) for centroids, clusters in zip(centroids_arr, clusters_arr)]

                new_clustering_img_arr = get_multi_comparable_colorings(clustering_img_arr[0], *clustering_img_arr[1:])
                unique_ref = np.unique(new_clustering_img_arr[0])
                colors = distinctipy.get_colors(len(unique_ref), pastel_factor=PASTEL_FACTOR)
                cm = distinctipy.get_colormap(colors)

                # add the other instance rhos

                for i, local_rho in enumerate(local_rhos):  
                    fig, ax = plt.subplots(figsize=(5,5), layout="tight")
                    ax.imshow(new_clustering_img_arr[i], cmap=cm)
                    ax.set_axis_off()
                    fig.savefig(record_paths[i], dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close()