import pathlib
from pathlib import Path
import glob
import os, sys

dir_name = '/scratch/elec/puhe/c/muse_2024/c2_muse_humor/feature_segments/hubert-superb/'
onlyfiles = glob.glob(f'{dir_name}/*/*.csv',  recursive = True)
onlyfiles = [f for f in onlyfiles if os.path.isfile(f)]
for _file in onlyfiles:
    spkr_id=Path(_file).parts[-2].split('_')[0]
    out_dir = os.path.join(dir_name, spkr_id)
    out_fname = Path(_file).parts[-1]
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    #print(_file, spkr_id, out_fname)
    os.rename(_file, os.path.join(out_dir, out_fname))