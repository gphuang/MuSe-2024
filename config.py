import os
from pathlib import Path
import torch
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# muse 2024 data
DATA_PATH = "/scratch/elec/puhe/c/muse_2024" 
PERCEPTION_PATH = os.path.join(DATA_PATH, 'c1_muse_perception')
HUMOR_PATH = os.path.join(DATA_PATH, 'c2_muse_humor')
BASE_PATH = "/scratch/work/huangg5/muse/MuSe-2024"  # "/home/lukas/Desktop/nas/data_work/LukasChrist/MuSe2024/"

PERCEPTION = 'perception'
HUMOR = 'humor'

TASKS = [PERCEPTION, HUMOR]

PATH_TO_FEATURES = {
    PERCEPTION: os.path.join(PERCEPTION_PATH, 'feature_segments'),
    HUMOR: os.path.join(HUMOR_PATH, 'feature_segments')
}

# humor is labelled every 2s, but features are extracted every 500ms
N_TO_1_TASKS = {HUMOR, PERCEPTION}

ACTIVATION_FUNCTIONS = {
    PERCEPTION: torch.nn.Sigmoid,
    HUMOR: torch.nn.Sigmoid
}

NUM_TARGETS = {
    HUMOR: 1,
    PERCEPTION: 1
}


PATH_TO_LABELS = {
    PERCEPTION: os.path.join(PERCEPTION_PATH, 'labels.csv'),
    HUMOR: os.path.join(HUMOR_PATH, 'label_segments')
}

PATH_TO_METADATA = {
    PERCEPTION: os.path.join(PERCEPTION_PATH, 'metadata'),
    HUMOR: os.path.join(HUMOR_PATH, 'metadata')
}

PARTITION_FILES = {task: os.path.join(path_to_meta, 'partition.csv') for task,path_to_meta in PATH_TO_METADATA.items()}

PERCEPTION_LABELS = ['admiring', 'aggressive', 'angry', 'arrogant', 'assertiv', 'attractive', 'charismatic', 'collaborative', 'compassionate', 'competent', 'competitive', 'confident', 'dominant', 'emotional', 'enthusiastic', 'envious', 'expressive', 'forceful', 'friendly', 'good_natured', 'independent', 'intelligent', 'kind', 'leader_like', 'likeable', 'naive', 'pity', 'productive', 'risk', 'sincere', 'sympathetic', 'trustworthy', 'understanding', 'warm', 'yielding']

current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:23]
OUTPUT_PATH = os.path.join(BASE_PATH, 'results')
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log_muse')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data_muse')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model_muse')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction_muse')
