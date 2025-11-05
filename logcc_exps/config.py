from pathlib import Path

ROOT_PATH = Path('/Users/chadepl/git/spatially-aware-correlation-clustering/')
DATA_DIR = ROOT_PATH.joinpath('experiments-data')


cat_dataset_map = {"cvp_meteo": "Meteo", "hptc_right_parotid": "HaN-ParotidR", "hptc_brainstem": "HaN-Brainstem", "hptc_right_parotid_3d": "HaN-ParotidR-3D", "hptc_brainstem_3d": "HaN-Brainstem-3D"}
cat_method_map = {"pfaffelmoser2012": "CN-Pivot", "pivot": "Pivot"}