# Standard library
from pathlib import Path

# Third-party
import pandas as pd

# Local modules
from logcc_exps.config import DATA_DIR
from logcc_exps.scripts.tables.tab_rd_times import load_df, process_df_categories, rename_df_categories

if __name__ == "__main__":
    
    data_dir = DATA_DIR / "rd_grid"
    df = load_df(data_dir, "times")
    df = process_df_categories(df)
    df = rename_df_categories(df)

    print(df)
    print()

    # we average the local timings
    df_local = df.copy()
    df_local = df_local.groupby(["dataset", "method", "accelerate_flag", "seed"], observed=True)["t_local"].mean().reset_index()
    df_local = df_local.groupby(["dataset", "method", "accelerate_flag"], observed=True)["t_local"].median().reset_index()

    # we sum the global timings
    df_global = df.copy()
    df_global = df_global.groupby(["dataset", "method", "accelerate_flag", "seed"], observed=True)["t_global"].sum().reset_index()
    df_global = df_global.groupby(["dataset", "method", "accelerate_flag"], observed=True)["t_global"].median().reset_index()

    # we compute total timings as the sum of the local and global averages 
    df_total = pd.merge(left=df_local, right=df_global, how="inner", left_on=["dataset", "method", "accelerate_flag"], right_on=["dataset", "method", "accelerate_flag"])
    df_total["t_total"] = df_total["t_local"] + df_total["t_global"]
    print(df_total)


    multi_tuples = [        
        ( True,  't_local'),
        ( True, 't_global'),
        ( True,  't_total'),
        (False,  't_total'),
        ]
    multi_cols = pd.MultiIndex.from_tuples(multi_tuples, names=["accelerated", "time_type"])

    df_pivoted = df_total.copy()
    df_pivoted = df_pivoted.pivot_table(index=["dataset", "method",], columns=["accelerate_flag",], values=["t_total", "t_local", "t_global"], observed=True).reset_index()
    df_pivoted = df_pivoted.set_index(["dataset", "method"])
    df_pivoted = df_pivoted.sort_index(axis=0, level=1)
    df_pivoted.index = df_pivoted.index.swaplevel()    
    df_pivoted.columns = df_pivoted.columns.swaplevel()    
    df_pivoted = pd.DataFrame(df_pivoted, columns=multi_cols)
    df_pivoted[("Speedup", "")] = df_pivoted[(False, "t_total")] / df_pivoted[(True, "t_total")]
    
    print(df_pivoted.index)
    print(df_pivoted.columns)
    print()
    print(df_pivoted)
    print(df_pivoted.to_latex(float_format="%.2f"))