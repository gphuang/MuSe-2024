import os, sys, re
import pandas as pd
import numpy as np 
import ast
import pathlib
from pathlib import Path
from matplotlib import pyplot as plt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # avoid xlabels to be cut off

model_types=('rnn', 'cnn', 'crnn', 'cnn-attn', 'crnn-attn')
features=('egemaps', 'ds', 'w2v-msp', 'faus', 'vit-fer', 'facenet512', 'hubert-superb', 'hubert-er')  
labels=('aggressive', 'arrogant', 'assertiv', 'confident', 'dominant', 'independent', 'risk', 'leader_like', 
        'collaborative', 'enthusiastic', 'friendly', 'good_natured', 'kind', 'likeable', 'sincere',  'warm') 

csv_in='results/csvs/perception.csv'
csv_path='results/csvs/perception_analysis.csv'

# output from script, missing performance for each of 5 seeds
df0 = pd.read_csv(csv_in, index_col=0)

# get Pearson from log file, output new csv 
if 0:
    os.remove(csv_path)
    for _model in model_types:
        for _feat in features:
            for _label in labels:
                _df = df0[(df0.feature==_feat) & (df0.label_dim==_label) & (df0.model_type==_model)]
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
                # (\d+\.\d+)
                # ^-?
                try:
                    tuples = re.findall(r'ID\/Seed\s(\d+)\s\|\sBest\s\[Val\sPearson\]\:\s*(-?\d+\.\d+)\s\|\s', text)
                    dct = {'val_Pearson': [float(v) for (k,v) in tuples]}
                    dct.update({'seed': [k for (k,v) in tuples]})
                except:
                    sys.exit(log_fname)
                dct.update({'model_type':_model})
                dct.update({'feature':_feat})
                dct.update({'label':_label})
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

# group by labels
df_out = df.groupby(['model_type', 'feature', 'label'])['val_Pearson'].agg([np.max, np.mean, np.std])
df_out.to_csv('results/csvs/perception_analysis_table2_labels.csv')
print(df_out)

# group by seeds
df_out = df.groupby(['model_type', 'feature'])['val_Pearson'].agg([np.max, np.mean, np.std])
df_out.to_csv('results/csvs/perception_analysis_table3_seeds.csv')
print(df_out)

# png
plot = df_out.plot(kind ='bar', title="Group by seeds")


ax = df_out.plot.bar()
ax.set_title("Group by seeds",color='black')
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.plot()
plt.tight_layout()
plt.savefig('results/pngs/perception_analysis_table3_seeds.png')
