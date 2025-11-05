# Standard library
from pathlib import Path
import pickle

# Third-party
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

# Local modules
from logcc_exps.config import DATA_DIR

FONT_SIZE = 15
FIGSIZE = (4, 4)

def load_df(path_to_records, glob_regex="[!.]*"):
    records = []
    for run_path in path_to_records.glob(glob_regex):
        with open(run_path, "rb") as f:
            print(f"Loading {f}")
            record = pickle.load(f)
            records.append(record)
            print(f" - Loaded")

    df = pd.DataFrame(records)
    return df


def plot_exp_field_size(df, method, reg_size=4, show=True):

    df_vis = df.copy()
    df_vis = df_vis[df_vis["method"] == method]
    df_vis = df_vis[np.logical_or(df_vis["clust_prop"] == 0.75, df_vis["clust_prop"] == 1.0)]
    df_vis = df_vis[df_vis["reg_size"] == reg_size]
    df_vis["clust_prop"] = df_vis["clust_prop"].cat.remove_unused_categories()

    df_vis_means = df_vis.groupby(["accelerate_flag", "sqrt_m", "clust_prop"])["time_secs"].mean().reset_index()    

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="tight")

    sns.scatterplot(df_vis_means, x="sqrt_m", y="time_secs", hue="accelerate_flag", style="clust_prop")
    sns.lineplot(df_vis, x="sqrt_m", y="time_secs", hue="accelerate_flag", style="clust_prop")

    ax.set_xlabel(r"$\sqrt{M}$=Rows=Cols", fontsize=FONT_SIZE)
    ax.set_ylabel("Log(Time (seconds))", fontsize=FONT_SIZE)
    ax.set_yscale('log')
    ax.get_legend().remove()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if show:
        ax.set_title(f"field-size ({method}; reg_size {reg_size}x{reg_size})")
        plt.show()
    else:
        fig.savefig(f"/Users/chadepl/Downloads/{method}-exp_scaling_field_size.png", dpi=300)

    # Speedup table
    print()
    print("plot_exp_field_size")
    print(f"Method: {method}")
    a = df_vis_means[np.logical_and(df_vis_means["clust_prop"] == 0.75, df_vis_means["accelerate_flag"] == False)]["time_secs"].values
    b = df_vis_means[np.logical_and(df_vis_means["clust_prop"] == 0.75, df_vis_means["accelerate_flag"] == True)]["time_secs"].values
    print(a)
    print(b)
    print(a/b)
    print()


def plot_exp_rho_t(df, method, show=True):

    df_vis = df.copy()
    df_vis = df_vis[df_vis["method"] == method]
    df_vis = df_vis[np.logical_or(df_vis["clust_prop"] == 0.75, df_vis["clust_prop"] == 1.0)]
    df_vis["clust_prop"] = df_vis["clust_prop"].cat.remove_unused_categories()

    df_vis_means = df_vis.groupby(["accelerate_flag", "num_reg", "clust_prop"])["time_secs"].mean().reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="tight")
    sns.scatterplot(df_vis_means, x="num_reg", y="time_secs", hue="accelerate_flag", style="clust_prop")
    sns.lineplot(df_vis, x="num_reg", y="time_secs", hue="accelerate_flag", style="clust_prop")
    ax.set_xlabel("Num local clusters", fontsize=FONT_SIZE)
    ax.set_ylabel("Log(Time (seconds))", fontsize=FONT_SIZE)
    ax.set_yscale('log')
    ax.get_legend().remove()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if show:
        ax.set_title(f"corr-threshold ({method})")
        plt.show()
    else:
        fig.savefig(f"/Users/chadepl/Downloads/{method}-exp_scaling_corr_thresh.png", dpi=300)

    # Speedup table
    print()
    print("plot_exp_rho_t")
    print(f"Method: {method}")
    a = df_vis_means[np.logical_and(df_vis_means["clust_prop"] == 0.75, df_vis_means["accelerate_flag"] == False)]["time_secs"].values
    b = df_vis_means[np.logical_and(df_vis_means["clust_prop"] == 0.75, df_vis_means["accelerate_flag"] == True)]["time_secs"].values
    print(a)
    print(b)
    print(a/b)
    print()


if __name__ == "__main__":
        
    data_dir = DATA_DIR / "scaling_grid_v1"
    SHOW = False

    METHODS = ["pivot", "pfaffelmoser2012"]

    exp_field_size = data_dir.joinpath("increasing_field_size")
    exp_corr_thresh = data_dir.joinpath("increasing_corr_thresh")
    
    df_fs = load_df(exp_field_size)
    df_ct = load_df(exp_corr_thresh)

    # Processing

    df_fs["method"] = pd.Categorical(df_fs["method"])
    df_fs["accelerate_flag"] = pd.Categorical(df_fs["accelerate_flag"])
    df_fs["clust_prop"] = pd.Categorical(df_fs["clust_prop"])

    df_ct["method"] = pd.Categorical(df_ct["method"])
    df_ct["accelerate_flag"] = pd.Categorical(df_ct["accelerate_flag"])
    df_ct["clust_prop"] = pd.Categorical(df_ct["clust_prop"])

    # PLOT: field size experiment
    
    for method in METHODS:
        plot_exp_field_size(df_fs, method, reg_size=4, show=SHOW)

    # PLOT: correlation threshold experiment

    for method in METHODS:
        plot_exp_rho_t(df_ct, method, show=SHOW)