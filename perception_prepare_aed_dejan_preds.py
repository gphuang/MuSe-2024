import os, sys
import pandas as pd
import numpy as np
import pathlib

from config import PREDICTION_FOLDER

fname='results/prediction_muse/perception/lf/label_devel.csv'
df_label=pd.read_csv(fname, index_col=0)
label_dims=df_label.columns
#print(df_label.shape, df_label.head(3))
#sys.exit(0)

def write_to_csv(df, csv_path):
    csv_dir = pathlib.Path(csv_path).parent.resolve()
    os.makedirs(csv_dir, exist_ok=True)

    df.to_csv(csv_path)
    print(f'Datafram written to {csv_path}.')

# prepare label wise predictions
"""
Input format:
subj_id,aggressive,arrogant,dominant,enthusiastic,friendly,leader_like,likeable,assertiv,confident,independent,risk,sincere,collaborative,kind,warm,good_natured
0,0.59383935,0.59933823,0.599389,0.59933823,0.5993886,0.599389,0.599389,0.5993886,0.599389,0.599389,0.599389,0.599389,0.59383935,0.599389,0.599389,0.599389
"""

"""
Output format:
devel partition
meta_col_0,prediction,label
9,0.39005035161972046,0.3528999984264374
10,0.42247816920280457,0.4943000078201294

test partition
meta_col_0,prediction,label
0,0.3390737771987915,
"""
if 0:
    for _label in label_dims:
        for partition in ['test', 'devel']: #
            in_fname=f'results/prediction_muse/perception/aed_dejan/predictions_{partition}.csv'
            df_pred=pd.read_csv(in_fname, index_col=0)
            #print(df_pred.shape, df_pred.head(3))
            _df=df_pred[[_label]]
            _df=_df.rename(columns={_label:'prediction'})
            #print(_df.shape, _df.head(3))
            #sys.exit(0)
            
            if partition=='devel':
                _df_label=df_label[[ _label]]
                _df_label=_df_label.rename(columns={_label:'label'})
                _df=pd.concat([_df, _df_label], axis=1)
            else:
                _df['label']=0

            _df.index.names=['meta_col_0']
            #print(_df.shape, _df.head(3))
            #sys.exit(0)

            out_fname=f'results/prediction_muse/perception/{_label}/aed_dejan/predictions_{partition}.csv'

            # write output
            write_to_csv(_df, out_fname)


# sbatch run-late-fusion-perception-gp-dejan.sh

# combine lf labels into 1 csv submission file
lf_dir='lf-gp-dajan'
for partition in ['devel', 'test']:
    appended_preds=[]
    for _label in label_dims[:]:
        pred_fname=os.path.join(PREDICTION_FOLDER, 'perception', _label, lf_dir, f'predictions_{partition}.csv')
        df_pred=pd.read_csv(pred_fname, index_col=0)
        #print(df_pred.shape, df_pred.head(2))
        df_pred=df_pred.rename(columns={'prediction': _label})
        appended_preds.append(df_pred)
    df_out=pd.concat(appended_preds, axis=1)
    df_out.index.names=['subj_id']
    #print(df_out.shape, df_out.head(3))

    # write output
    csv_path=os.path.join(PREDICTION_FOLDER, 'perception', lf_dir, f'predictions_{partition}.csv')
    write_to_csv(df_out, csv_path)
    
# cp results/prediction_muse/perception/lf/*.csv results/submission/c1_perception




