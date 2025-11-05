# Third-party
import matplotlib.pyplot as plt
import napari
import numpy as np

# Local modules
from logcc_exps.lib.clustering.logcc import (
    global_step_pivot,
    local_step,
)
from logcc_exps.lib.clustering.utils import (
    get_clustering_img,
    get_comparable_clustering_colorings,
)
from logcc_exps.lib.data.synthetic_data import abc_blobs

if __name__ == "__main__":
    ensemble = abc_blobs(100, 512, iso_value=None, seed=42)
    CORR_FN = lambda a, b: np.corrcoef(a, b)[0, 1]

    local_centroids, local_clusters = local_step(ensemble, rho=0.99, corr_fn=CORR_FN)
    global_centroids, global_clusters = global_step_pivot(ensemble, local_centroids, local_clusters, rho=0.5, corr_fn=CORR_FN)

    local_clustering = get_clustering_img(ensemble, local_centroids, local_clusters)
    global_clustering = get_clustering_img(ensemble, global_centroids, global_clusters)

    clust1, clust2 = get_comparable_clustering_colorings(ensemble, local_clustering, global_clustering)
    print(clust1.shape)

    fig, ax = plt.subplots(figsize=(5,5), layout="tight")
    ax.imshow(ensemble.mean(axis=0), cmap="gray")
    ax.set_axis_off()
    fig.savefig("/Users/chadepl/Downloads/blobs_ensemble_mean.png", dpi=300)

    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
    ax.imshow(clust1, cmap="rainbow")
    ax.set_axis_off()
    fig.savefig(f"/Users/chadepl/Downloads/blob_local-clustering.png", dpi=300, bbox_inches='tight', pad_inches=0.0)

    fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
    ax.imshow(clust2, cmap="rainbow")
    ax.set_axis_off()
    fig.savefig(f"/Users/chadepl/Downloads/blob_global-clustering.png", dpi=300, bbox_inches='tight', pad_inches=0.0)

    # 1. Some samples of the ensemble (viridis color)
    for i in np.random.choice(np.arange(ensemble.shape[0]), 10, replace=False):
        fig, ax = plt.subplots(figsize=(5, 5), layout="tight")
        ax.imshow(ensemble[i], cmap="gray")
        ax.set_axis_off()
        fig.savefig(f"/Users/chadepl/Downloads/blob-ensemble_member-{i}.png", dpi=300, bbox_inches='tight', pad_inches=0.0)

    

    # viewer = napari.view_image(ensemble)
    # viewer.add_labels(local_clustering)
    # viewer.add_labels(global_clustering)
    # napari.run()