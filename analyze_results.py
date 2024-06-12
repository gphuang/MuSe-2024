import os, sys
import pandas as pd
import numpy as np 
import ast

model_types=('cnn', 'rnn', 'crnn', 'cnn-attn', 'crnn-attn')
labels=('aggressive' , 'arrogant' , 'dominant' , 'enthusiastic' , 'friendly' , 'leader_like' , 'likeable' , 'assertiv' , 'confident' , 'independent' , 'risk' , 'sincere' , 'collaborative' , 'kind' , 'warm' , 'good_natured') 
features= ('faus', 'facenet512', 'vit-fer', 'w2v-msp', 'egemaps --normalize', 'ds')  # ('hubert-superb', 'hubert-er') # 

csv='results/csvs/perception.csv'
df = pd.read_csv(csv, index_col=0)
print(df.shape)
if 0:
    _out = df.groupby(['feature', 'model_type'])['best_val_Pearson'].agg([np.mean, np.std])
    print(_out)

for _model in model_types:
    for _feat in features:
        for _label in labels:
            _df = df[(df.feature==_feat) & (df.label_dim==_label) & (df.model_type==_model)]
            _dict = ast.literal_eval(_df['paths'].values[0])
            model_name = os.path.basename(_dict['model'])
            log_fname = os.path.join(_dict['log'], model_name + '.txt')
            print(model_name, log_fname)
            assert os.path.exists(log_fname)
            # "ID/Seed 101 | Best [Val Pearson]: 0.1720 |  Loss: ..."
            # seed: 
            # val_pearson:
            sys.exit(0)

