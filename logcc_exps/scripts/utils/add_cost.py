# Third-party
import numpy as np
import pandas as pd
from sklearn import metrics

# Local modules
from logcc_exps.lib.clustering.utils import get_clustering_img


def table_cost_data(ensemble, clustering_records, data_dir, fn_stem=""):

    csv_path = data_dir.joinpath(f"{fn_stem}.csv")
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        N, ROWS, COLS = ensemble.shape
        M = ROWS * COLS

        df_records = []
        clustering_imgs = []
        for record in clustering_records:
            
            centroids = record["global_centroids"]
            clusters = record["global_clusters"]            
            clustering_imgs.append(get_clustering_img(ensemble, centroids, clusters))

            df_records.append({
                "dataset": record["dataset"],
                "method": record["method"],
                "accelerate_flag": record["accelerate_flag"],
                "local_rho": record["local_rho"],
                "rho": record["rho"],
                "seed": record["seed"],
                "pos_edge_error_sum": 0,
                "neg_edge_error_sum": 0,
            })

        for i in range(M):
            print(f"Processing location: {i}")
            ri, ci = divmod(i, COLS)
            for j in range(i + 1, M):
                rj, cj = divmod(j, COLS)

                ei = ensemble[:, ri, ci]
                ej = ensemble[:, rj, cj]

                corr = np.corrcoef(ei, ej)[0, 1]

                for df_record, clustering_img in zip(df_records, clustering_imgs):
                    
                    record_rho = df_record["rho"]
                    is_link_pos = clustering_img[ri, ci] == clustering_img[rj, cj]

                    if corr >= record_rho and not is_link_pos:
                        df_record["pos_edge_error_sum"] += 1 # pos_edge_error
                    if corr < record_rho and is_link_pos:
                        df_record["neg_edge_error_sum"] += 1 # neg_edge_error


        df = pd.DataFrame(df_records)
        df.to_csv(csv_path)

    return df


def compute_quality_metrics(ref_clusterings, acc_clusterings, return_mean=False):
    # These metrics do not make sense for Pivot because there is a lot of variation within results (both accelerated and un-accelerated)
    # Therefore, comparing accelerated vs un accelerated does not tell us much. In summary, Pivot is highly sensitive to the starting point.
    # For Pfaffelmoser it might make more sense given that it is deterministic. In this case, it might be worth it
    # evaluating separately with and without disconnected components as this might affect the results. 
    # Also in Pfaffelmoser, many pixels are left un-assigned, which might add noise to the result. Perhaps a threshold could help
    # filtering out components that do not add much.
    K_ref = len(ref_clusterings)
    K_acc = len(acc_clusterings)
    out_metrics = {"reg_rand":[], "adj_rand":[], "adj_mi":[], "norm_mi":[]}
    for i in range(K_ref):        
        for j in range(i + 1, K_acc):
            labels_ref = ref_clusterings[i]
            labels_acc = acc_clusterings[j]
            out_metrics["reg_rand"].append(metrics.rand_score(labels_ref, labels_acc))
            out_metrics["adj_rand"].append(metrics.adjusted_rand_score(labels_ref, labels_acc))
            out_metrics["adj_mi"].append(metrics.adjusted_mutual_info_score(labels_ref, labels_acc))
            out_metrics["norm_mi"].append(metrics.normalized_mutual_info_score(labels_ref, labels_acc))
    if return_mean:
        for k, v in out_metrics.items():
            out_metrics[k] = np.median(np.array(v))
    return out_metrics   