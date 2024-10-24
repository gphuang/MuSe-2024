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

model_types=('rnn' 'cnn' 'crnn' 'cnn-attn' 'crnn-attn') #('rnn') #
_labels=('aggressive' 'arrogant' 'dominant' 'enthusiastic' 'friendly' 'leader_like' 'likeable' 'assertiv' 'confident' 'independent' 'risk' 'sincere' 'collaborative' 'kind' 'warm' 'good_natured') #('aggressive') # 
labels=('attractive' 'charismatic' 'competitive' 'expressive' 'naive')
features=('w2v-msp' 'egemaps' 'ds' 'vit-fer' 'faus' 'facenet512' 'bert-base-uncased' 'bert-base-cased' 'bert-base-multilingual-uncased' 'bert-base-multilingual-cased' 'avhubert-base-lrs3-iter5' 'hubert-superb' 'roberta-base' 'gpt2') #('ds') #
impression_lengths=(1 2 3 4 5) #(1 2) # 
impression_positions=('random' 'last' 'first') #('random') #  

# RNN
num_rnn_layers=2
model_dims=(512)

# GENERAL
lrs=(0.0005)
patience=10
n_seeds=3 # 5
dropouts=(0.4)
batch_size=32

# adapt
csv='results/csvs/perception.csv'

for model_type in "${model_types[@]}"; do
    for feature in "${features[@]}"; do
        for impression_length in "${impression_lengths[@]}"; do
            for impression_position in "${impression_positions[@]}"; do
                for model_dim in "${model_dims[@]}"; do
                    for lr in "${lrs[@]}";do
                        for dropout in "${dropouts[@]}";do
                            for label in "${labels[@]}"; do
                                python3 main.py --task perception --feature "$feature" --impression_length "$impression_length" --impression_position "$impression_position" --batch_size $batch_size --model_type $model_type --model_dim $model_dim --label_dim "$label" --rnn_bi --rnn_n_layers $num_rnn_layers --lr "$lr" --n_seeds "$n_seeds" --linear_dropout $dropout --rnn_dropout $dropout --early_stopping_patience 10 --predict --result_csv "$csv" # --save_ckpt #
                            done
                            done
                        done
                    done
                done
            done
        done
    done

