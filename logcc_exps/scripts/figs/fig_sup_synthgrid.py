# Standard library
from pathlib import Path

# Third-party
import distinctipy
import matplotlib.pyplot as plt
import napari
import numpy as np

# Local modules
from logcc_exps.lib.clustering.pivot import correlation_clustering
from logcc_exps.lib.clustering.utils import get_clustering_img
from logcc_exps.lib.data.grids import (
    correlation_grid_aggregation,
    correlation_grid_division,
)

if __name__ == "__main__":
    # we use distinctpy for the colors
    print("Method")

    REG_SIZE = 4
    

    # centroids, clusters = correlation_clustering(ensemble, 0.5, lambda a, b: np.corrcoef(a,b)[0,1], seed=42)
    # cluster_img = get_clustering_img(ensemble, centroids, clusters)

    if False:

        prop_clusters = [0.1, 0.5, 1.0][2]

        out_path = Path(f"/Users/chadepl/Downloads/synthgrid/agg_{prop_clusters}")
        out_path.mkdir(parents=True, exist_ok=True)    

        ensembles = []
        clustering_imgs = []
        colors = []
        cms = []

        for num_reg in [4, 8, 12, 16]:#[20, 24, 28, 32]:
            run_out_path = out_path.joinpath(f"num-reg-{num_reg}")
            run_out_path.mkdir(parents=True, exist_ok=True)

            print(np.ceil((num_reg * num_reg)*prop_clusters))

            ensemble, cluster_img_gt = correlation_grid_aggregation(100,
                                                                    REG_SIZE, REG_SIZE, 
                                                                    num_reg, num_reg, np.ceil((num_reg * num_reg)*prop_clusters), 
                                                                    return_cluster_img=True, seed=42)
            
            unique_labs = np.unique(cluster_img_gt)
            choices = np.arange(unique_labs.size * 10)
            np.random.shuffle(choices)
            for i, ul in enumerate(unique_labs):
                cluster_img_gt[cluster_img_gt == ul] = choices[i] # reshuffle colors

            color = distinctipy.get_colors(unique_labs.size)
            cm = distinctipy.get_colormap(color)

            ensembles.append(ensemble)
            clustering_imgs.append(cluster_img_gt)
            colors.append(color)
            cms.append(cm)

            fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
            ax.imshow(cluster_img_gt, cmap=cm)
            ax.set_axis_off()
            fig.savefig(run_out_path.joinpath("clustering-gt.png"), dpi=300, bbox_inches='tight', pad_inches=0.0)

            for i in range(10):
                fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
                ax.imshow(ensemble[i], cmap="gray")
                ax.set_axis_off()
                fig.savefig(run_out_path.joinpath(f"ensemble_member-{i}.png"), dpi=300, bbox_inches='tight', pad_inches=0.0)

    # viewer = napari.view_image(ensembles[0])
    # viewer.add_image(clustering_imgs[0], colormap=cms[0])
    # # viewer.add_image(cluster_img)
    # napari.run()

    if True:

        prop_clusters = [0.1, 0.5, 1.0][2]

        out_path = Path(f"/Users/chadepl/Downloads/synthgrid/div_{prop_clusters}")
        out_path.mkdir(parents=True, exist_ok=True)    

        ensembles = []
        clustering_imgs = []
        colors = []
        cms = []

        for num_reg in [4, 8, 12, 16]:#[20, 24, 28, 32]:
            run_out_path = out_path.joinpath(f"num-reg-{num_reg}")
            run_out_path.mkdir(parents=True, exist_ok=True)

            print(np.ceil((num_reg * num_reg)*prop_clusters))

            ensemble, cluster_img_gt = correlation_grid_division(100,
                                                                128, 128,
                                                                num_reg, num_reg, np.ceil((num_reg * num_reg)*prop_clusters), 
                                                                return_cluster_img=True, seed=42)
            
            unique_labs = np.unique(cluster_img_gt)
            choices = np.arange(unique_labs.size * 10)
            np.random.shuffle(choices)
            for i, ul in enumerate(unique_labs):
                cluster_img_gt[cluster_img_gt == ul] = choices[i] # reshuffle colors

            color = distinctipy.get_colors(unique_labs.size)
            cm = distinctipy.get_colormap(color)

            ensembles.append(ensemble)
            clustering_imgs.append(cluster_img_gt)
            colors.append(color)
            cms.append(cm)

            fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
            ax.imshow(cluster_img_gt, cmap=cm)
            ax.set_axis_off()
            fig.savefig(run_out_path.joinpath("clustering-gt.png"), dpi=300, bbox_inches='tight', pad_inches=0.0)

            for i in range(10):
                fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
                ax.imshow(ensemble[i], cmap="gray")
                ax.set_axis_off()
                fig.savefig(run_out_path.joinpath(f"ensemble_member-{i}.png"), dpi=300, bbox_inches='tight', pad_inches=0.0)