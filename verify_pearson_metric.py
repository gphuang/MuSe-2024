import os, sys
import pandas as pd
import numpy as np
import glob

from eval import calc_pearsons

dir_results='/scratch/work/huangg5/muse/MuSe-2024/results/prediction_muse/'
task='perception'
label_dim='aggressive/RNN_2024-07-03-14-49_[vit-fer]_[0.0005_32]'
label_dims=[
    'aggressive/RNN_2024-07-03-14-49_[vit-fer]_[0.0005_32]',
    'arrogant/RNN_2024-07-03-14-50_[vit-fer]_[0.0005_32]',
    'assertiv/RNN_2024-07-03-14-58_[vit-fer]_[0.0005_32]',
    'collaborative/RNN_2024-07-03-15-06_[vit-fer]_[0.0005_32]',
    'confident/RNN_2024-07-03-15-00_[vit-fer]_[0.0005_32]',
    'dominant/RNN_2024-07-03-14-52_[vit-fer]_[0.0005_32]',
    'enthusiastic/RNN_2024-07-03-14-53_[vit-fer]_[0.0005_32]',
    'friendly/RNN_2024-07-03-14-54_[vit-fer]_[0.0005_32]',
    'good_natured/RNN_2024-07-03-15-10_[vit-fer]_[0.0005_32]',
    'independent/RNN_2024-07-03-15-01_[vit-fer]_[0.0005_32]',
    'kind/RNN_2024-07-03-15-07_[vit-fer]_[0.0005_32]',
    'leader_like/RNN_2024-07-03-14-55_[vit-fer]_[0.0005_32]',
    'likeable/RNN_2024-07-03-14-57_[vit-fer]_[0.0005_32]',
    'risk/RNN_2024-07-03-15-03_[vit-fer]_[0.0005_32]',
    'sincere/RNN_2024-07-03-15-04_[vit-fer]_[0.0005_32]',
    'warm/RNN_2024-07-03-15-09_[vit-fer]_[0.0005_32]',
]

if 0:
    f_name=os.path.join(dir_results, task, label_dim, 'predictions_devel.csv')
    df=pd.read_csv(f_name, index_col=0)
    print(df.shape, df.head(3))
    preds=df['prediction'].to_numpy()
    labels=df['label'].to_numpy()
    print(preds.shape, labels.shape)
    p = calc_pearsons(preds, labels)
    print(p)

#TBD walk in dir and collect latest pred.
label_dims=('aggressive', 'confident', 'good_natured',) 
label_dims+=('arrogant', 'assertiv',  'dominant', 'independent', 'risk', 'leader_like', 'collaborative', 'enthusiastic', 'friendly', 'kind', 'likeable', 'sincere',  'warm') 
feat_type, model_type=('vit', 'crnn-attn') #('egemaps', 'iaf') #('faus', 'iaf') #('vit', 'rnn') #  

#TBD: labels 16
p_values=[]
for label_dim in label_dims:
    prediction_dir=os.path.join(dir_results, task, label_dim)
    onlyfiles = glob.glob(f'{prediction_dir}/{model_type.upper()}*{feat_type}*/*devel.csv',  recursive = True)
    onlyfiles = [f for f in onlyfiles if os.path.isfile(f)]
    assert len(onlyfiles)>=1
    f_name=onlyfiles[-1] # os.path.join(dir_results, task, _dim, 'predictions_devel.csv')
    df=pd.read_csv(f_name, index_col=0)
    preds=df['prediction'].to_numpy()
    labels=df['label'].to_numpy()
    p = calc_pearsons(preds, labels)
    p_values.append(p)
    print(f_name)
    #print(df.shape, preds.shape, labels.shape, p)
assert len(p_values) == len(label_dims)
print(np.mean(p_values), p_values)
#TBD: RNN vs. IAF

