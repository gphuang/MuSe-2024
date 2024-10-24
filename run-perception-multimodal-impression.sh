#!/bin/bash
#SBATCH --time=49:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load mamba

source activate muse

model_types=('iaf') # ('lmf' 'tfn' 'iaf') #
_labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured') #('aggressive') #  
labels=('attractive' 'charismatic' 'competitive' 'expressive' 'naive')
audio_features=('w2v-msp' 'hubert-superb' 'avhubert-base-lrs3-iter5') #('w2v-msp') #
video_features=('faus' 'facenet512' 'vit-fer') #('vit-fer') #
text_features=('bert-base-cased' 'bert-base-multilingual-uncased') #('bert-base-uncased' 'bert-base-multilingual-cased' 'roberta-base' 'xlm-roberta-large' 'gpt2')  #('bert-base-uncased' 'bert-base-multilingual-cased') #('bert-base-uncased') #
impression_lengths=(1 2 3 4 5) #(1 2) # 
impression_positions=('random' 'last' 'first') #('random') #  

# GENERAL
lr=0.0005
patience=10
n_seeds=3 #5
dropout=0.4
batch_size=32

# adapt
csv='results/csvs/perception_fusion.csv'

for model_type in "${model_types[@]}"; do
    for impression_length in "${impression_lengths[@]}"; do
        for impression_position in "${impression_positions[@]}"; do
            for a_feature in "${audio_features[@]}"; do
                for v_feature in "${video_features[@]}";do
                    for t_feature in "${text_features[@]}";do
                        for label in "${labels[@]}"; do
                            list="$a_feature $v_feature $t_feature"   
                            echo ${list}
                            python3 main.py --task perception --feature "$list" --impression_length "$impression_length" --impression_position "$impression_position" --batch_size $batch_size --model_type "$model_type" --label_dim "$label" --lr "$lr" --n_seeds "$n_seeds" --early_stopping_patience 10 --predict --result_csv "$csv" # --save_ckpt 
                        done
                        done
                    done
                done
            done
        done
    done

