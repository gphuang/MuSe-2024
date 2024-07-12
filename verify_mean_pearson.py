import os, sys
import pandas as pd
import numpy as np

import config
from eval import mean_pearsons
from data_parser import load_data
from dataset import MultiModalMuSeDataset, MuSeDataset

fname='./results/prediction_muse/perception/lf/predictions_devel.csv' # 0.5409033566745889
fname='./results/prediction_muse/perception/top_1/predictions_devel.csv' # 0.5300840487431339
fname='./results/prediction_muse/perception/pseudo_fusion/predictions_devel.csv' # 0.5073189186614222
df_pred=pd.read_csv(fname, index_col=0)
#print(df_pred.shape)

fname='results/prediction_muse/perception/lf/label_devel.csv'
df_label=pd.read_csv(fname, index_col=0)
#print(df_label.shape, df_label.head(3))

label_arrays=df_label.values
pred_arrays=df_pred.values # .iloc[:, 1:]
print(pred_arrays.shape, label_arrays.shape)
result = mean_pearsons(pred_arrays, label_arrays)
print(result) 