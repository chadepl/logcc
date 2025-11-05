"""
Script that generates the tables with times and CC scores.
For the CC scores, it computes the correlation matrix online to prevent crashing with bigger datasets.
Before running this script, make sure you generated the data with the script `data_rd_grid.py`
"""
# Standard library
from pathlib import Path
import pickle

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Local modules
from logcc_exps.config import DATA_DIR
from logcc_exps.scripts.utils.add_time import table_times_data
from logcc_exps.scripts.utils.add_cost import table_cost_data
                

if __name__ == "__main__":

    data_dir = DATA_DIR / "data/rd_grid/"

    METHOD = ["pivot", "pfaffelmoser2012"][1]
    RHOS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    dataset_name = ["cvp_meteo", "hptc_right_parotid", "hptc_brainstem"][2]
    dataset_path = data_dir.joinpath(dataset_name)
    
    if not dataset_path.exists():
        raise Exception("No data")
    
    ensemble = np.load(dataset_path.joinpath("ensemble.npy"))
    N, ROWS, COLS = ensemble.shape

    #####################
    # Filtering records #
    #####################

    clustering_records = []
    for d in dataset_path.glob("*.pkl"):
        with open(d, "rb") as f:
            record = pickle.load(f)

            if record["method"] == METHOD:
                print(record["dataset"], record["method"], record["rho"], record["accelerate_flag"], record["t_local"], record["t_global"], record["t_total"])

                clustering_records.append(record)
                
 

    df_times = table_times_data(ensemble, clustering_records, f"{dataset_path.stem}-{METHOD}-times")
    df_cfv = table_cost_data(ensemble, clustering_records, f"{dataset_path.stem}-{METHOD}-cfv")

    df_cfv["total_error_sum"] = df_cfv["pos_edge_error_sum"] + df_cfv["neg_edge_error_sum"]

    print(df_times)
    print(df_cfv)

    # charts
    sns.boxplot(df_times, x="rho", y="t_total", hue="accelerate_flag")
    plt.show()

    sns.boxplot(df_cfv, x="rho", y="total_error_sum", hue="accelerate_flag")
    plt.show()
    sns.boxplot(df_cfv, x="rho", y="pos_edge_error_sum", hue="accelerate_flag")
    plt.show()
    sns.boxplot(df_cfv, x="rho", y="neg_edge_error_sum", hue="accelerate_flag")
    plt.show()

    # Times table


    # Cost function values table


    
    