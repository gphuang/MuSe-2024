import os, sys, re
import pandas as pd
import numpy as np 
import ast
import pathlib
from pathlib import Path
from matplotlib import pyplot as plt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # avoid xlabels to be cut off

model_types = ('rnn', ) 
model_types += ('crnn', 'cnn', 'cnn-attn', 'crnn-attn',)
features = ('egemaps', 'w2v-msp', 'ds', 'hubert-superb') 
features += ('faus', 'facenet512', 'vit-fer') 
features += ('bert-base-uncased', 'bert-base-multilingual-cased', 'roberta-base', 'xlm-roberta-large', 'gpt2')
features_av = ('avhubert-base-lrs3-iter5', 'avhubert-large-lrs3-iter5', 'avhubert-base-vox-iter5', 'avhubert-large-vox-iter5', 'avhubert-base-noise-pt-noise-ft-30h', 'avhubert-large-noise-pt-noise-ft-30h')
features_fusion = ('permute(a+v+t)')
label_dims = ('aggressive',) 
label_dims += ('arrogant', 'assertiv', 'confident', 'dominant', 'independent', 'risk', 'leader_like', 'collaborative', 'enthusiastic', 'friendly', 'good_natured', 'kind', 'likeable', 'sincere',  'warm') 

csv_in='results/csvs/perception_crnn_attn.csv'
csv_in='results/csvs/perception_avt.csv'
csv_in='results/csvs/perception_avt_fusion.csv'
csv_path='results/csvs/perception_analysis.csv'
metric = 'best_val_Pearson'

# verify baseline table2 & table3 numbers
# https://docs.google.com/spreadsheets/d/1qR8YCxBqnx5o9RiGvDO76w3EZx4PI2v71fe_nQg1rL4/edit?usp=sharing

df0 = pd.read_csv(csv_in, index_col=0)
print(df0.shape, df0.head(3))
# TBD get feat-model-label combos
sys.exit(0)

# get Pearson from log file, output new csv, performance for each of 5 seeds
if 1:
    if os.path.exists(csv_path):
        os.remove(csv_path)
    for _model in set(model_types):
        for _feat in set(features):
            for _label in label_dims:
                _df = df0[(df0.feature==_feat) & (df0.label_dim==_label) & (df0.model_type==_model)]
                #TBD: 19June updated runs that produce duplicated or updated results in df.
                #find experiments that have been interrupted
                print(_model, _feat, _label, _df.shape)
                _dict = ast.literal_eval(_df['paths'].values[0])
                model_name = os.path.basename(_dict['model'])
                log_fname = os.path.join(_dict['log'], model_name + '.txt')
                assert os.path.exists(log_fname)
                # Open and read the file.
                with open(log_fname, 'rt', encoding='utf-8') as f:
                    text = f.read()
                # "ID/Seed 101 | Best [Val Pearson]: 0.1720 |  Loss: ..."
                # "ID/Seed 101 | Best [Val Pearson]:-0.0965 | Loss:..."
                # Extract all the data tuples with a findall() each tuple is: (seed, Best Val Pearson)
                try:
                    tuples = re.findall(r'ID\/Seed\s(\d+)\s\|\sBest\s\[Val\sPearson\]\:\s*(-?\d+\.\d+)\s\|\s', text)
                    dct = {metric: [float(v) for (k,v) in tuples]}
                    dct.update({'seed': [k for (k,v) in tuples]})
                except:
                    sys.exit(log_fname)
                dct.update({'model_type':_model})
                dct.update({'feature':_feat})
                dct.update({'label_dim':_label})
                df = pd.DataFrame(dct)

                # make sure the directory exists
                csv_dir = pathlib.Path(csv_path).parent.resolve()
                os.makedirs(csv_dir, exist_ok=True)

                # write back
                if os.path.exists(csv_path):
                    old_df = pd.read_csv(csv_path)
                    df = pd.concat([old_df, df])
                df.to_csv(csv_path, index=False)

df = pd.read_csv(csv_path)

# TBD: which column to aggregate. For all three options, best feat-model combo remains the same.
# https://docs.google.com/spreadsheets/d/1qR8YCxBqnx5o9RiGvDO76w3EZx4PI2v71fe_nQg1rL4/edit?gid=0#gid=0
if 1:
    # group by best_seeds      
    df1 = df.groupby(['model_type', 'feature', 'label_dim']).agg(max_p=(metric, 'max'))           
    df2 = df1.groupby(['model_type', 'feature']).agg({'max_p': ['max', 'mean', 'std']})
    print(df2)

if 0:
    # group by seeds
    df1 = df.groupby(['model_type', 'feature', 'label_dim']).agg(mean_p=(metric, 'mean'))
    df2 = df1.groupby(['model_type', 'feature']).agg({'mean_p': ['max', 'mean', 'std']})
    print(df2)

if 0:
    # group by labels
    df1 = df.groupby(['model_type', 'feature', 'seed']).agg(mean_p=(metric, 'mean'))
    df2 = df1.groupby(['model_type', 'feature']).agg({'mean_p': ['max', 'mean', 'std']})
    print(df2)

df1.to_csv('results/csvs/table2_perception.csv')
df2.to_csv('results/csvs/table3_perception.csv')


if 0:
    # png
    plot = df2.plot(kind ='bar', title="Group by seeds")
    ax = df2.plot.bar()
    ax.set_title("Group by seeds",color='black')
    ax.legend(bbox_to_anchor=(1.0, 1.0))
    ax.plot()
    plt.tight_layout()
    plt.savefig('results/pngs/perception_analysis.png')

