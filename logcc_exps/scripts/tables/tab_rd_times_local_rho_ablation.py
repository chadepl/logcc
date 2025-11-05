# Standard library
from pathlib import Path
import pickle

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Local modules
from logcc_exps.scripts.tables.tab_rd_times import load_df, process_df_categories, rename_df_categories
from logcc_exps.config import DATA_DIR, cat_dataset_map, cat_method_map


if __name__ == "__main__":
    SHOW = False
    METHOD = list(cat_method_map.values())[0]    
    DATASETS = list(cat_dataset_map.values())
    RHOS = [0.5, 0.6, 0.7, 0.8, 0.9]

    # Load dataset from scaling experiment on real data (gives us the times of the unaccelerated variant)
    df_rd_scaling = load_df(DATA_DIR / "rd_grid_v1", which="times")
    df_rd_scaling = process_df_categories(df_rd_scaling)
    df_rd_scaling = rename_df_categories(df_rd_scaling)
    df_rd_scaling = df_rd_scaling[[a in RHOS for a in df_rd_scaling["rho"]]]
    df_rd_scaling["rho"] = pd.Categorical(df_rd_scaling["rho"], categories=RHOS)

    df_rd_scaling_acc_true = df_rd_scaling[df_rd_scaling["accelerate_flag"]==True]
    df_rd_scaling_acc_true["local_rho"] = 0.99
    df_rd_scaling_acc_true = df_rd_scaling_acc_true.loc[:, ["method", "dataset", "local_rho", "rho", "seed", "t_total"]]
    df_rd_scaling_acc_false = df_rd_scaling[df_rd_scaling["accelerate_flag"]==False].loc[:, ["method", "dataset", "rho", "seed", "t_total"]]
    df_rd_scaling_acc_false["local_rho"] = 0
    df_rd_scaling_acc_false = df_rd_scaling_acc_false.loc[:, ["method", "dataset", "local_rho", "rho", "seed", "t_total"]]

    df_rd_scaling_acc_true = df_rd_scaling_acc_true.groupby(["method", "dataset", "local_rho", "rho"], observed=False)["t_total"].mean()    
    df_rd_scaling_acc_true = df_rd_scaling_acc_true.reset_index()
    df_rd_scaling_acc_true = df_rd_scaling_acc_true.set_index(["method", "dataset", "rho"])
    df_rd_scaling_acc_true = df_rd_scaling_acc_true.drop("local_rho", axis=1)
    df_rd_scaling_acc_true = df_rd_scaling_acc_true.rename(dict(t_total="t_total_acc_true"), axis=1)

    df_rd_scaling_acc_false = df_rd_scaling_acc_false.groupby(["method", "dataset", "local_rho", "rho"], observed=False)["t_total"].mean()    
    df_rd_scaling_acc_false = df_rd_scaling_acc_false.reset_index()
    df_rd_scaling_acc_false = df_rd_scaling_acc_false.set_index(["method", "dataset", "rho"])
    df_rd_scaling_acc_false = df_rd_scaling_acc_false.drop("local_rho", axis=1)
    df_rd_scaling_acc_false = df_rd_scaling_acc_false.rename(dict(t_total="t_total_acc_false"), axis=1)


    # Load dataset from ablation experiment (gives us times using different local thresholds)
    data_dir = DATA_DIR / "rd_grid_local_rhos"
    df_ablation = load_df(data_dir, which="times")
    df_ablation = process_df_categories(df_ablation)
    df_ablation = rename_df_categories(df_ablation)
    
    df_ablation = df_ablation.loc[:, ["method", "dataset", "local_rho", "rho", "seed", "t_total"]]
    df_ablation = df_ablation[[a in RHOS for a in df_ablation["rho"]]]
    df_ablation["rho"] = pd.Categorical(df_ablation["rho"], categories=RHOS)

    df_ablation = df_ablation.groupby(["method", "dataset", "local_rho", "rho"], observed=False)["t_total"].mean()    
    df_ablation = df_ablation.reset_index()
    df_ablation = df_ablation.set_index(["method", "dataset", "rho"])

    df_ablation_99 = df_ablation[df_ablation["local_rho"] == 0.99].drop("local_rho", axis=1).rename(dict(t_total="t_total_0.99"), axis=1)
    df_ablation_07 = df_ablation[df_ablation["local_rho"] == 0.7].drop("local_rho", axis=1).rename(dict(t_total="t_total_0.07"), axis=1)
    df_ablation_05 = df_ablation[df_ablation["local_rho"] == 0.5].drop("local_rho", axis=1).rename(dict(t_total="t_total_0.05"), axis=1)
    

    df_all_times = pd.concat([
        df_rd_scaling_acc_false,
        df_rd_scaling_acc_true,
        df_ablation_99,
        df_ablation_07,
        df_ablation_05], axis=1) #.join(, ["method", "dataset", "rho"], how="left")

    print(df_all_times)


    df_speedups = df_all_times.copy()
    df_speedups["speedup_true"] = df_speedups["t_total_acc_false"]/df_speedups["t_total_acc_true"]
    df_speedups["speedup_0.99"] = df_speedups["t_total_acc_false"]/df_speedups["t_total_0.99"]
    df_speedups["speedup_0.07"] = df_speedups["t_total_acc_false"]/df_speedups["t_total_0.07"]
    df_speedups["speedup_0.05"] = df_speedups["t_total_acc_false"]/df_speedups["t_total_0.05"]
    df_speedups = df_speedups.drop(["t_total_acc_false", "t_total_acc_true", "t_total_0.99", "t_total_0.07", "t_total_0.05"], axis=1)

    df_speedups_mean = df_speedups.groupby(["method", "dataset"]).mean().drop(["speedup_true"], axis=1)

    print(df_speedups)
    print(df_speedups_mean)

    print()
    print(df_speedups_mean.reset_index().to_latex(float_format="%.2f"))

   