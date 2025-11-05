"""
This script tests the sensitivity of the final clustering with respect to different values of local rho and acc/no acc.
"""

from time import time
import numpy as np

import sys
sys.path.insert(0, "..")
from src.data.meteo_data import get_cvp_meteo_data
from src.data.han_data import hptc_brainstem

from src.clustering.pivot import correlation_clustering as pivot
from src.clustering.logcc import local_step, global_step_pivot, correlation_clustering, CCMethod

from src.clustering.utils import get_clustering_img


RHOS_USER = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
RHOS_GLOBAL = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
RHOS_LOCAL = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

CORR_FN = lambda ei, ej: np.corrcoef(ei, ej)[0, 1]
SEEDS = [42, 67, 98, 21, 10, 11, 87, 45, 98, 38] # ten trial

METHOD=CCMethod.CN_PIVOT

ensemble = get_cvp_meteo_data(scale_factor=1)
# ensemble = hptc_brainstem(scale_factor=1, iso_value=None, only_ensemble=True)
N, ROWS, COLS = ensemble.shape
M = ROWS * COLS

if __name__ == "__main__":

    exp_dict = {}

    for rho_user in RHOS_USER:
        exp_dict[f"{rho_user}"] = {}

        for rho_global in RHOS_GLOBAL:
            print(f"Running exp rho_t={rho_global}")

            exp_dict[f"{rho_user}"][f"{rho_global}"] = {}
        
            print(f" - Processing key noacc")        
            exp_dict[f"{rho_user}"][f"{rho_global}"]["noacc"] = {}
            for seed in SEEDS:
                print(f" -- Processing seed {seed}")
                t_start = time()
                centroids, clusters = pivot(ensemble, rho_global, CORR_FN, seed)
                t_end = time()
                t_total = t_end - t_start
                exp_dict[f"{rho_user}"][f"{rho_global}"]["noacc"][seed] = {
                    "cimg": get_clustering_img(ensemble, centroids, clusters), 
                    "time": t_total,
                    "pos_counts": 0,
                    "neg_counts": 0,
                    }

            for rho_local in RHOS_LOCAL:
                variant_name = f"acc-{rho_local}"
                print(f" - Processing key {variant_name}")
                exp_dict[f"{rho_user}"][f"{rho_global}"][variant_name] = {}
                for seed in SEEDS:
                    print(f" -- Processing seed {seed}")
                    t_start = time()
                    centroids, clusters = correlation_clustering(ensemble, rho_global, CORR_FN, rho_local, method=METHOD, seed=seed)
                    t_end = time()
                    t_total = t_end - t_start
                    exp_dict[f"{rho_user}"][f"{rho_global}"][variant_name][seed] = {
                        "cimg": get_clustering_img(ensemble, centroids, clusters), 
                        "time": t_total,
                        "pos_counts": 0,
                        "neg_counts": 0,
                        }
                

    for i in range(M):
        print(f"Processing location: {i}")
        ri, ci = divmod(i, COLS)
        for j in range(i + 1, M):
            rj, cj = divmod(j, COLS)

            ei = ensemble[:, ri, ci]
            ej = ensemble[:, rj, cj]

            corr = np.corrcoef(ei, ej)[0, 1]

            for rho_user, rho_user_dict in exp_dict.items():
                rho_threshold = float(rho_user)
                for rho_global, rho_global_dict in rho_user_dict.items():                
                    for variant_name, variant_dict in rho_global_dict.items():
                        for seed_number, seed_dict in variant_dict.items():
                            cimg = seed_dict["cimg"]
                            is_link_pos = cimg[ri, ci] == cimg[rj, cj]

                            if corr >= rho_threshold and not is_link_pos:
                                seed_dict["pos_counts"] += 1 # pos_edge_error
                            if corr < rho_threshold and is_link_pos:
                                seed_dict["neg_counts"] += 1 # neg_edge_error

    # Build dict
    records = []
    for rho_user, rho_user_dict in exp_dict.items():
        rho_threshold = float(rho_user)
        for rho_global, rho_global_dict in rho_user_dict.items():                
            for variant_name, variant_dict in rho_global_dict.items():
                for seed_number, seed_dict in variant_dict.items():
                    record = {
                        "rho_user": rho_user,
                        "rho_global": rho_global,
                        "variant": variant_name,
                        "seed": seed_number, 
                        "time": seed_dict['time'], 
                        "pos_counts": seed_dict['pos_counts'], 
                        "neg_counts": seed_dict['neg_counts'], 
                        "total_counts": seed_dict['pos_counts'] + seed_dict['neg_counts']
                    }
                    records.append(record)

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv("data_rd_local_rho_sensitivity_meteo.csv")
    print("Success")

    # print("[")
    # for dk, dv in exp_dict.items():
    # # for i in range(len(outs)):
    #     for dv_key, dv_value in dv.items():
    #         record = f"""
    #                 'version': '{dk}', 
    #                 'seed': {dv_key}, 
    #                 'time': {dv_value['time']}, 
    #                 'pos_counts': {dv_value['pos_counts']}, 
    #                 'neg_counts': {dv_value['neg_counts']}, 
    #                 'total_counts': {dv_value['pos_counts'] + dv_value['neg_counts']}
    #             """
    #         record = "{" + record + "},"
    #         print(record)
    # print("]")

