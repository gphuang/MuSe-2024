import os, sys
import pandas as pd
import numpy as np

from eval import mean_pearsons

fname='./results/prediction_muse/perception/pseudo_fusion/predictions_devel.csv'
df=pd.read_csv(fname)
print(df.shape, df.head(10))

filter_col=[col for col in df if col.endswith('_label')]
df_label=df[filter_col]
filter_col=[col for col in df if not col.endswith('_label')]
df_pred=df[filter_col ]

label_arrays=df_label.values
pred_arrays=df_pred.iloc[:, 1:].values
print(pred_arrays.shape, label_arrays.shape)
result = mean_pearsons(pred_arrays, label_arrays)
print(result) # 0.5073189186614222