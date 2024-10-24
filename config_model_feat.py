# 35 perception labels 
from config import PERCEPTION_LABELS
# 16 labels on aggentive and communal scale
label_dims=['aggressive', 'arrogant', 'dominant', 'enthusiastic', 'friendly', 
            'leader_like', 'likeable', 'assertiv', 'confident', 'independent', 
            'risk', 'sincere', 'collaborative', 'kind', 'warm', 'good_natured']
# 21 annotated labels 
label_dims+=['attractive', 'charismatic', 'competitive', 'expressive', 'naive']

# impression position and length
_positions=[ 'random', 'first', 'last']
_max_len=6 #11 # impression_length
_feat_lens=['-'.join([_position, str(i), 'sec']) for _position in _positions for i in range(1, _max_len) ] 

# feat-model reps of uni-modal
_model_types=['CNN', 'CRNN', 'RNN', 'CNN-ATTN', 'CRNN-ATTN']
_features=['vit-fer', 'w2v-msp', 'hubert-superb'] # 'bert-multilingual' for Humor
_features+=['avhubert-base-lrs3-iter5']
_features+=['bert-base-uncased', 'bert-base-cased', 'bert-base-multilingual-uncased', 'bert-base-multilingual-cased', 'roberta-base', 'xlm-roberta-large', 'gpt2'] #
_features+=['faus', 'facenet512', 'egemaps', 'ds'] 
_impressions=['+'.join([_feat, _len]) for _feat in _features for _len in _feat_lens]
_hyper='0.0005_32'
unimodals=[_model+'_['+_feat+']_['+_hyper+']' for _model in _model_types for _feat in _features]
unimodals+=[_model+'_['+_feat+']_['+_hyper+']' for _model in _model_types for _feat in _impressions]
#print(len(unimodals)) 

# feat-model reps of multi-modal
_model_types=['IAF', 'LMF', 'TFN'] #'LMF'
_a_types=[ 'egemaps', 'ds', 'w2v-msp', 'hubert-superb', 'avhubert-base-lrs3-iter5']
_v_types=['vit-fer', 'faus', 'facenet512']
_t_types=['bert-base-uncased', 'bert-base-cased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'roberta-base', 'xlm-roberta-large', 'gpt2'] # bert-multilingual is for Humor 
_features=['+'.join([_a, _v, _t,]) for _a in _a_types for _v in _v_types for _t in _t_types]
_impressions=['+'.join([_feat, _len]) for _feat in _features for _len in _feat_lens]
_hyper='0.0005_32'
multimodals=[_model+'_['+_feat+']_['+_hyper+']' for _model in _model_types for _feat in _features]
multimodals+=[_model+'_['+_feat+']_['+_hyper+']' for _model in _model_types for _feat in _impressions]
#print(len(multimodals)) 

dct_fusion_models={
#fname='./results/prediction_muse/perception/top_1/predictions_devel.csv', # 0.5326608133674763 # same with with vit-vit-vit
'late_fusion+impressions': './results/prediction_muse/perception/lf+impressions/predictions_devel.csv', # 0.48671576204462974
'late_fusion': './results/prediction_muse/perception/lf-gp-dejan/predictions_devel.csv', # 0.5472685244138151  
'late_fusion_25':'./results/prediction_muse/perception/lf/predictions_devel.csv', # 0.5429725331490107 # 0.5409033566745889 with with vit-vit-vit
'late_fusion_impressions-5sec': './results/prediction_muse/perception/lf-impressions-5-sec/predictions_devel.csv', # 0.4454161000650498
'pseudo_fusion+impressions': './results/prediction_muse/perception/pseudo_fusion+impressions/predictions_devel.csv', # 0.528863484305494
'pseudo_fusion':'./results/prediction_muse/perception/pseudo_fusion-gp-dejan/predictions_devel.csv', # 0.5201279332221813
'psedu_fusion_25':'./results/prediction_muse/perception/pseudo_fusion/predictions_devel.csv', # 0.5073189186614222 # same with with vit-vit-vit
'pseudo_fusion_impressions-5sec': './results/prediction_muse/perception/pseudo_fusion-impressions-5-sec/predictions_devel.csv', # 0.4799345309593144
'AED':'./results/prediction_muse/perception/aed_dejan/predictions_devel.csv', # 0.2710262891342781
}