# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Local modules
from logcc_exps.scripts.tables.tab_rd_times import (
    load_df,
    process_df_categories,
    rename_df_categories,
)
from logcc_exps.config import DATA_DIR, cat_dataset_map, cat_method_map

FONT_SIZE = 12

if __name__ == "__main__":
    
    SHOW = False
    DATASETS = list(cat_dataset_map.values())
    data_dir = DATA_DIR / "rd_grid"
    ERROR_TYPE = ["total_edge_error_sum", "pos_edge_error_sum", "neg_edge_error_sum"][0]

    df = load_df(data_dir, which="cfv")
    df = process_df_categories(df)
    df = rename_df_categories(df)

    df["total_edge_error_sum"] = df["pos_edge_error_sum"] + df["neg_edge_error_sum"]

    for dataset_name in DATASETS:
        df_boxplot = df.copy()
        df_boxplot = df_boxplot[df_boxplot["dataset"] == dataset_name]

        df_boxplot_means = df_boxplot.groupby(["method", "rho", "accelerate_flag"], observed=True)[ERROR_TYPE].mean().reset_index()

        fig, ax = plt.subplots(figsize=(4,3), layout="tight")
        sns.scatterplot(df_boxplot_means, x="rho", y=ERROR_TYPE, hue="method", style="accelerate_flag", legend=None, ax=ax)
        sns.lineplot(df_boxplot, x="rho", y=ERROR_TYPE, hue="method", style="accelerate_flag", legend=None, ax=ax)
        ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        ax.set_ylabel("Cost function value", fontsize=FONT_SIZE)
        ax.set_xlabel(r"Correlation threshold ($\rho_t$)", fontsize=FONT_SIZE)
        
        if SHOW:
            plt.show()
        else:
            fig.savefig(f"/Users/chadepl/Downloads/cfv-{dataset_name}.png", dpi=300)

    
    # Granular visualization of the different error components
    if True:
        print("All errors overview")
        METHOD = list(cat_method_map.values())[1]
        print(f" - {METHOD}")

        for dataset_name in DATASETS:
            df_boxplot = df.copy()
            df_boxplot = df_boxplot[df_boxplot["dataset"] == dataset_name]
            df_boxplot = df_boxplot[df_boxplot["method"] == METHOD]
            error_types = ["total_edge_error_sum", "pos_edge_error_sum", "neg_edge_error_sum"]

            df_boxplot = pd.melt(df_boxplot, id_vars=["dataset", "method", "accelerate_flag", "rho", "seed"], value_vars=error_types, var_name="error_type", value_name="error_value")
            df_boxplot["error_type"] = pd.Categorical(df_boxplot["error_type"])

            fig, ax = plt.subplots()

            df_boxplot_means = df_boxplot.groupby(["method", "rho", "accelerate_flag", "error_type"], observed=True)["error_value"].mean().reset_index()        
            sns.scatterplot(df_boxplot_means, x="rho", y="error_value", hue="error_type", style="accelerate_flag", legend=True, ax=ax)
            sns.lineplot(df_boxplot, x="rho", y="error_value", hue="error_type", style="accelerate_flag", legend=False, ax=ax)

            ax.set_xticks([0.3, 0.5, 0.7, 0.9])
            ax.set_ylabel("Cost function value")
            ax.set_xlabel(r"$\rho_t$")
            
            if SHOW:
                plt.show()
            else:
                fig.savefig(f"/Users/chadepl/Downloads/cfv-{dataset_name}-allcosts.png", dpi=300)
