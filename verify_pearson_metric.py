import os, sys
import pandas as pd

from eval import calc_pearsons

dir_results='/scratch/work/huangg5/muse/MuSe-2024/results/prediction_muse/'
task='perception'
label_dim='aggressive/RNN_2024-07-03-14-49_[vit-fer]_[0.0005_32]'
#label_dim='arrogant/RNN_2024-07-03-14-50_[vit-fer]_[0.0005_32]'
f_name=os.path.join(dir_results, task, label_dim, 'predictions_devel.csv')
df=pd.read_csv(f_name, index_col=0)
print(df.shape, df.head(3))
preds=df['prediction'].to_numpy()
labels=df['label'].to_numpy()
print(preds.shape, labels.shape)
p = calc_pearsons(preds, labels)
print(p)

