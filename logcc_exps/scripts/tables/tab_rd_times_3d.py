# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Local modules
from logcc_exps.config import DATA_DIR, cat_method_map

cat_dataset_map = {"hptc_right_parotid_3d": "HaN-ParotidR-3D", "hptc_brainstem_3d": "HaN-Brainstem-3D"}

def load_df(data_dir, which="times"):
    dfs = []
    for times_csv_path in data_dir.glob(f"*{which}*"):
        print(times_csv_path)
        dfs.append(pd.read_csv(times_csv_path))

    df = pd.concat(dfs, axis=0).reset_index()
    df = df.drop(df.columns[:2], axis=1) # first two columns are artifacts
    return df

def process_df_categories(df, rhos=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    df["dataset"] = pd.Categorical(df["dataset"], categories=cat_dataset_map.keys())
    df["rho"] = pd.Categorical(df["rho"], categories=rhos)
    df["method"] = pd.Categorical(df["method"], categories=cat_method_map.keys())
    df["accelerate_flag"] = pd.Categorical(df["accelerate_flag"], categories=[False, True])
    return df

def rename_df_categories(df):
    # Renaming categories
    df["dataset"] = df["dataset"].cat.rename_categories(cat_dataset_map)
    df["method"] = df["method"].cat.rename_categories(cat_method_map)
    # df["accelerate_flag"] = df["accelerate_flag"].cat.rename_categories({False: , True: "Pfaffelmoser2012"})
    return df

if __name__ == "__main__":
    data_dir = DATA_DIR / "rd_grid_3d_fullvol_pivot"

    df = load_df(data_dir, which="times")
    df = process_df_categories(df)

    RHO = 0.9

    # Filtering
    df = df[df["rho"].apply(lambda rho: rho in [RHO])]
    print(df.head())

    df = rename_df_categories(df)

    df_means = df.groupby(["dataset", "rho", "method", "accelerate_flag"], observed=True)["t_total"].median().reset_index() # median for now because outliers affect
    #df_stds = df.groupby(["dataset", "rho", "method", "accelerate_flag"])["t_total"].std().reset_index()
    df_pivoted_means = df_means.pivot_table(index=["dataset", "method"], columns=["rho", "accelerate_flag"], values="t_total", observed=True).reset_index()
    #df_pivoted_stds = df_stds.pivot_table(index=["dataset", "rho"], columns=["method", "accelerate_flag"], values="t_total").reset_index()

    # speedups
    df_pivoted_means[("speedup", "")] = df_pivoted_means[(RHO, False)]/df_pivoted_means[(RHO, True)]

    print(df_pivoted_means)
    print()
    print(df_pivoted_means.to_latex(float_format="%.2f"))
    
