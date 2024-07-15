import os, sys
import pandas as pd
import pathlib

from config import PREDICTION_FOLDER

label_dims = ('aggressive', 'confident', 'good_natured',) 
label_dims += ('arrogant', 'assertiv',  'dominant', 'independent', 'risk', 'leader_like', 'collaborative', 'enthusiastic', 'friendly', 'kind', 'likeable', 'sincere',  'warm') 

criteria=sys.argv[1]#'pseudo_fusion'#'top_1'#'positive_pearsons'# 
fname='./results/csvs/table2_pred_perception.csv'
df=pd.read_csv(fname)
df=df.sort_values(['label_dim', 'mean_pearsons'], ascending=False)
# print(df.shape, df.columns, df.head(3))

# select model_ids from overall performance ranking
"""fname='./results/csvs/table3_pred_perception.csv'
df0=pd.read_csv(fname)
df0=df0.sort_values(['mean_pearsons'], ascending=False)
df1=df0.head(20)"""
# models rep of uni and multi-modal, for pseudo-fusion, as late_fusion.py not working for c1
_model_types=['CNN', 'CRNN', 'RNN', 'CNN-ATTN', 'CRNN-ATTN']
_feat_types=['vit-fer', 'w2v-msp', 'hubert-superb']
_hyper='0.0005_32'
unimodals=[_model+'_['+_feat+']_['+_hyper+']' for _model in _model_types for _feat in _feat_types]
print(len(unimodals)) # 15

_model_types=['LMF', 'IAF']
_a_types=['w2v-msp', 'hubert-superb']
_v_types=['vit-fer', ]
_t_types=['bert-base-uncased', 'bert-base-multilingual-cased', 'roberta-base', 'xlm-roberta-large', 'gpt2']  
_feat_types=[ _a+'_'+_v+'_'+_t for _a in _a_types for _v in _v_types for _t in _t_types ]
#_feat_types+=['vit-fer_vit-fer_vit-fer']
_hyper='0.0005_32'
multimodals=[_model+'_['+_feat+']_['+_hyper+']' for _model in _model_types for _feat in _feat_types]
print(len(multimodals)) # 11

top_models=unimodals+multimodals
print(len(top_models)) # 25

if 0:
    top_models=[
    'CNN-ATTN_[vit-fer]_[0.0005_32]',
    'CRNN_[vit-fer]_[0.0005_32]',
    'CNN_[vit-fer]_[0.0005_32]',
    'CRNN-ATTN_[vit-fer]_[0.0005_32]',
    'RNN_[vit-fer]_[0.0005_32]',
    'IAF_[vit-fer_vit-fer_vit-fer]_[0.0005_32]',
    'IAF_[w2v-msp_vit-fer_gpt2]_[0.0005_32]', 
    'IAF_[w2v-msp_vit-fer_bert-base-multilingual-cased]_[0.0005_32]',
    'IAF_[w2v-msp_vit-fer_bert-base-uncased]_[0.0005_32]',
    'IAF_[w2v-msp_vit-fer_roberta-base]_[0.0005_32]',
    'IAF_[hubert-superb_vit-fer_bert-base-multilingual-cased]_[0.0005_32]',
    'IAF_[hubert-superb_vit-fer_bert-base-uncased]_[0.0005_32]',
    'IAF_[hubert-superb_vit-fer_roberta-base]_[0.0005_32]',
    'LMF_[vit-fer_vit-fer_vit-fer]_[0.0005_32]',
    'LMF_[w2v-msp_vit-fer_gpt2]_[0.0005_32]', 
    'LMF_[w2v-msp_vit-fer_bert-base-multilingual-cased]_[0.0005_32]',
    'LMF_[w2v-msp_vit-fer_bert-base-uncased]_[0.0005_32]',
    'LMF_[w2v-msp_vit-fer_roberta-base]_[0.0005_32]',
    'LMF_[hubert-superb_vit-fer_bert-base-multilingual-cased]_[0.0005_32]',
    'LMF_[hubert-superb_vit-fer_bert-base-uncased]_[0.0005_32]',
    'LMF_[hubert-superb_vit-fer_roberta-base]_[0.0005_32]',
    ]

for partition in ['devel', 'test']:
    appended_preds=[]
    for _label in label_dims[:]:
        _df=df[df.label_dim==_label]
        if criteria=='positive_pearsons':
            _df=_df[_df.mean_pearsons>0]
        elif criteria.startswith('top'):
            k=int(criteria.split('_')[-1])
            #_df=_df.loc[_df.mean_pearsons.idxmax()]
            _df=_df.iloc[k-1] # Series result # df.iloc[[1]]  # DataFrame result
        else:
            # 'pseudo_fusion':
            # avt features or the fusion of them. include v-v-v
            _df=_df[_df['model_id'].isin(top_models)]
            _df=_df.iloc[0]
            
        log_name=_df.log_name
        pred_fname=os.path.join(PREDICTION_FOLDER, 'perception', _label, log_name, f'predictions_{partition}.csv')
        df_pred=pd.read_csv(pred_fname, index_col=0)
        df_pred=df_pred.rename(columns={'prediction': _label})
        #print(df_pred.shape, df_pred.head(3))
        # save to label dir, to do late fusion, 'label' is kept for late_fusion.py
        csv_path=os.path.join(PREDICTION_FOLDER, 'perception', _label, criteria, f'predictions_{partition}.csv')
        csv_dir = pathlib.Path(csv_path).parent.resolve()
        os.makedirs(csv_dir, exist_ok=True)
        df_pred.to_csv(csv_path)
        # append to df and save to perception dir
        #if 0 and partition =='devel': # keep label for sanity check
        #    df_pred=df_pred.rename(columns={'label': _label+'_label'})
        #else:
        df_pred=df_pred.drop(columns=['label'])
        appended_preds.append(df_pred)
    df_out=pd.concat(appended_preds, axis=1)
    df_out.index.names=['subject_id']
    #print(df_out.shape, df_out.head(3))
    #sys.exit(0)

    # write output
    csv_path=os.path.join(PREDICTION_FOLDER, 'perception', criteria, f'predictions_{partition}.csv')
    csv_dir = pathlib.Path(csv_path).parent.resolve()
    os.makedirs(csv_dir, exist_ok=True)

    df_out.to_csv(csv_path)
    print(f'Prediction file written to {csv_path}.')
    
# cp results/prediction_muse/perception/lf/*.csv results/submission/c1_perception