import pandas as pd

def table_times_data(ensemble, clustering_records, data_dir, fn_stem=""):

    csv_path = data_dir.joinpath(f"{fn_stem}.csv")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        if len(ensemble.shape) == 3:
            N, ROWS, COLS = ensemble.shape
            M = ROWS * COLS        
        else:
            N, SLICES, ROWS, COLS = ensemble.shape
            M = SLICES * ROWS * COLS

        df_records = []
        
        for record in clustering_records:
            df_records.append({
                "dataset": record["dataset"],
                "method": record["method"],
                "accelerate_flag": record["accelerate_flag"],
                "local_rho": record["local_rho"],
                "rho": record["rho"],
                "seed": record["seed"],
                "t_local": record["t_local"],
                "t_global": record["t_global"],
                "t_total": record["t_total"]
            })
        
        df = pd.DataFrame(df_records)
        df.to_csv(csv_path)
    
    return df