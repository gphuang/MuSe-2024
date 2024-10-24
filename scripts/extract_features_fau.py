# alternative facial action units with pyfeat
# https://py-feat.org/basic_tutorials/05_fex_analysis.html#extract-facial-features-using-detector

# Uncomment the line below and run this only if you're using Google Collab
# !pip install py-feat
# source activate ser_venv

import os, sys
from tqdm import tqdm
from feat import Detector

detector=Detector(emotion_model='svm')

# Loop over and process each video and save results to csv
OVERWRITE=True

#demo
video='/scratch/work/huangg5/tutorials/data_example/clip.mp4'
out_name='/scratch/work/huangg5/tutorials/data_example/fau-feat.csv'
if not os.path.exists(out_name) and OVERWRITE:
    print(f"Processing: {video}")
    fex=detector.detect_video(video)
    print(f'fex.shape: {fex.shape}')
    fex.to_csv(out_name, index=False)

# muse provided fau