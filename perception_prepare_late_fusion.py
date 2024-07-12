import os, sys
import pandas as pd
import pathlib

from config import PREDICTION_FOLDER

label_dims = ('aggressive', 'confident', 'good_natured',) 
label_dims += ('arrogant', 'assertiv',  'dominant', 'independent', 'risk', 'leader_like', 'collaborative', 'enthusiastic', 'friendly', 'kind', 'likeable', 'sincere',  'warm') 

fname='./results/csvs/table2_pred_perception.csv'
df=pd.read_csv(fname)

# 37 models top and rep of uni and multi-modal, for pseudo-fusion, as late_fusion.py not working for c1
_model_types=['CNN', 'CRNN', 'RNN', 'CNN-ATTN', 'CRNN-ATTN']
_feat_types=['w2v-msp', 'hubert-superb', 'vit-fer',]
_hyper='0.0005_32'
unimodals=[_model+'_['+_feat+']_['+_hyper+']' for _model in _model_types for _feat in _feat_types]
print(len(unimodals))

_model_types=['IAF'] # 'LMF' only exists for some labels, it performs well on 'assertive'
_a_types=['w2v-msp', 'hubert-superb']
_v_types=['vit-fer', ]
_t_types=['bert-base-uncased', 'bert-base-multilingual-cased', 'roberta-base', 'xlm-roberta-large', 'gpt2'] 
_av_types=['avhubert-base-lrs3-iter5', 'avhubert-large-lrs3-iter5', 'avhubert-base-vox-iter5', 'avhubert-large-vox-iter5', 'avhubert-base-noise-pt-noise-ft-30h', 'avhubert-large-noise-pt-noise-ft-30h'] 
_feat_types=[ _a+'_'+_v+'_'+_t for _a in _a_types for _v in _v_types for _t in _t_types ] # avt combos
_feat_types+=['vit-fer_vit-fer_vit-fer']
_hyper='0.0005_32'
multimodals=[_model+'_['+_feat+']_['+_hyper+']' for _model in _model_types for _feat in _feat_types]
print(len(multimodals))
top_models=unimodals+multimodals
print(len(top_models))

# prepare sbatch script
if 0:
    cmds=[]
    for _label in label_dims[:]:
        _df=df[df.label_dim==_label]
        _df=_df[_df['model_id'].isin(top_models)]
        #print(_df.shape, ' '.join(_df.log_name.values))
        _ids= ' '.join(_df.log_name.values)
        _cmd=f'python3 late_fusion.py --task perception --submission_format --result_csv $csv --label_dim {_label} --model_ids {_ids}'
        cmds.append(_cmd)

    # output
    # python3 late_fusion.py --task perception --submission_format --label_dim aggressive --model_ids top_1 top_2 top_3 top_4 top_5
    header_txt="""#!/bin/bash
    #SBATCH --time=23:59:59
    #SBATCH --partition=gpu-h100-80g 
    #SBATCH --gpus=1
    #SBATCH --mem=250G
    #SBATCH --cpus-per-task=6
    #SBATCH --output=logs/%A.out
    #SBATCH --job-name=muse
    #SBATCH -n 1

    module load mamba

    source activate muse

    echo "late fusion."

    ## perception
    csv='perception_late_fusion.csv'

    """
    with open('./run-late-fusion-perception.sh', 'w') as f:
        f.write(f"{header_txt}")
        for line in cmds:
            f.write(f"{line}\n")
    # sbatch run-late-fusion-perception.sh

# combine lf labels into 1 csv submission file
for partition in ['devel', 'test']:
    appended_preds=[]
    for _label in label_dims[:]:
        pred_fname=os.path.join(PREDICTION_FOLDER, 'perception', _label, 'lf', f'predictions_{partition}.csv')
        df_pred=pd.read_csv(pred_fname, index_col=0)
        #print(df_pred.shape, df_pred.head(2))
        df_pred=df_pred.rename(columns={'prediction': _label})
        appended_preds.append(df_pred)
    df_out=pd.concat(appended_preds, axis=1)
    df_out.index.names=['subject_id']
    #print(df_out.shape, df_out.head(3))

    # write output
    csv_path=os.path.join(PREDICTION_FOLDER, 'perception', 'lf', f'predictions_{partition}.csv')
    csv_dir = pathlib.Path(csv_path).parent.resolve()
    os.makedirs(csv_dir, exist_ok=True)

    df_out.to_csv(csv_path)
    print(f'Prediction file written to {csv_path}.')
    
# cp results/prediction_muse/perception/lf/*.csv results/submission/c1_perception
