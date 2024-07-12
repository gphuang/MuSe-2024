import os, sys
import pandas as pd
import pathlib

from config import PREDICTION_FOLDER

label_dims = ('aggressive', 'confident', 'good_natured',) 
label_dims += ('arrogant', 'assertiv',  'dominant', 'independent', 'risk', 'leader_like', 'collaborative', 'enthusiastic', 'friendly', 'kind', 'likeable', 'sincere',  'warm') 

criteria=sys.argv[1] #'top_1'# 'positive_pearsons' #
fname='./results/csvs/table2_pred_perception.csv'
df=pd.read_csv(fname)
df=df.sort_values(['label_dim', 'mean_pearsons'], ascending=False)
print(df.shape, df.columns, df.head(3))

for partition in ['devel', 'test']:
    appended_preds=[]
    for _label in label_dims[:]:
        _df=df[df.label_dim==_label]
        if criteria=='positive_pearsons':
            _df=_df[_df.mean_pearsons>0]
        if criteria.startswith('top'):
            k=int(criteria.split('_')[-1])
            #_df=_df.loc[_df.mean_pearsons.idxmax()]
            _df=_df.iloc[k-1] # Series result # df.iloc[[1]]  # DataFrame result
            #print(_df.model_id, _df.shape, _df)
            log_name=_df.log_name
            pred_fname=os.path.join(PREDICTION_FOLDER, 'perception', _label, log_name, f'predictions_{partition}.csv')
            df_pred=pd.read_csv(pred_fname, index_col=0)
            df_pred=df_pred.rename(columns={'prediction': _label})
            #print(df_pred.shape, df_pred.head(3))
            # save to label dir, try to do late fusion
            csv_path=os.path.join(PREDICTION_FOLDER, 'perception', _label, criteria, f'predictions_{partition}.csv')
            csv_dir = pathlib.Path(csv_path).parent.resolve()
            os.makedirs(csv_dir, exist_ok=True)
            df_pred.to_csv(csv_path)
            # save to perception dir
            df_pred=df_pred.drop(columns=['label'])
            appended_preds.append(df_pred)
    df_out=pd.concat(appended_preds, axis=1)
    #print(df_out.shape, df_out.head(3))
    #sys.exit(0)

    # write output
    csv_path=os.path.join(PREDICTION_FOLDER, 'perception', criteria, f'predictions_{partition}.csv')
    csv_dir = pathlib.Path(csv_path).parent.resolve()
    os.makedirs(csv_dir, exist_ok=True)

    df_out.to_csv(csv_path)
    print(f'Prediction file written to {csv_path}.')
    
sys.exit(0)
      
# output: 
# python3 late_fusion.py --task perception --result_csv $csv --label_dim warm --model_ids IAF_2024-07-11-04-33_[w2v-msp_avhubert-large-noise-pt-noise-ft-30h_vit-fer]_[0.0005_32] 
