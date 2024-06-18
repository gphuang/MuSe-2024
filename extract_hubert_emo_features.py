# extract emotional features from audio using HuBert
# source: https://huggingface.co/superb/hubert-large-superb-er

import sys, os
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
import torch
import torch.nn as nn
import librosa
import datasets
from datasets import load_dataset
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

def rolling_window(a, window, step):
    """
    Reshape a numpy array 'a' of shape (n) to form shape((n - window_size) // step + 1, window_size))

    Create a function to reshape a 1d array using a sliding window with a step.
    NOTE: The function uses numpy's internat as_strided function because looping in python is slow in comparison.

    Adopted from http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html and 
    https://gist.github.com/codehacken/708f19ae746784cef6e68b037af65788

    """
    shape = a.shape[:-1] + ((a.shape[-1] - window + 1)//step, window)
    strides = (a.strides[0] * step,) + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# muse perception baseline features
if 0:
    """
    time_step, feat_dim]
    win_len=2000ms, hop_len=500ms, audio_len=30s
    30/0.5=60
    0 w2v-msp (58, 1025)
    0 egemaps (58, 89)
    0 vit-fer (62, 769)
    1 w2v-msp (57, 1025)
    1 egemaps (57, 89)
    1 vit-fer (61, 769)
    """
    for spkr in [0, 1, 10, 100]:
        for feat in ['w2v-msp', 'egemaps', 'vit-fer']:
        
            fname = f'/scratch/elec/puhe/c/muse_2024/c1_muse_perception/feature_segments/{feat}/{spkr}.csv'
            df = pd.read_csv(fname, index_col=0)
            print(spkr, feat, df.shape)
    sys.exit(0)

# configs
model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
audio_data = '/scratch/elec/puhe/c/muse_2024/c1_muse_perception/raw/wav'

# hubert-superb-er demo
def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example
if 0:
    # load a demo dataset and read audio files
    dataset = load_dataset("anton-l/superb_demo", "er", split="session1")
    dataset = dataset.map(map_to_array)

    # compute attention masks and normalize the waveform if needed
    inputs = feature_extractor(dataset[:4]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
    print(inputs)

    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]

    print(logits.shape, predicted_ids.shape)
    sys.exit(0)

# hubert-superb-er feature extration on muse
if 0:
    """
    logits: [bs, emo_classes]
    model.config.label2id: {
        "ang": 2,
        "hap": 1,
        "neu": 0,
        "sad": 3
    },
    model(**inputs): logits
    """
    # dataset object huggingface
    dataset = load_dataset("audiofolder", data_dir=audio_data, drop_labels=True, drop_metadata=True)
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    audio_arrays = [x["array"] for x in dataset['train']["audio"]]
    audio_lens =[len(_array) for _array in audio_arrays[:10]]
    print(audio_lens)

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(audio_data) if isfile(join(audio_data, f))]
for _file in onlyfiles:
    spkr_id = Path(_file).stem
    print(f'Spkr: {spkr_id}')
    input_fname = os.path.join(audio_data, str(spkr_id) + '.wav')
    input_audio, sample_rate = librosa.load(input_fname,  sr=16000)

    # segment audio with win_len=2000ms, hop_len=500ms, audio_len=30s
    audio_arrays = rolling_window(input_audio, window=int(2*16000), step=int(0.5*16000))
    audio_lens =[_array.shape for _array in audio_arrays]

    # extract features from layers
    inputs = feature_extractor(audio_arrays, sampling_rate=16000, padding=True, return_tensors="pt")
    outputs = model(**inputs) # odict_keys(['logits', 'hidden_states'])
    logits = outputs.logits # torch.Size([57, 4])
    _states = outputs.hidden_states  # tuple, size=25, 
    last_hidden_state = outputs.hidden_states[-1] # torch.Size([56, 99, 1024])
    print(model, last_hidden_state.shape)
    # averaging rep. in final layer
    activations = torch.mean(last_hidden_state, 1)

    # save features (logits_emo4, h_states512, h_states1024) for each audio file hubert-superb-er
    feat_dir = '/scratch/elec/puhe/c/muse_2024/c1_muse_perception/feature_segments' 
    
    out_dir = os.path.join(feat_dir, 'hubert-superb')
    out_fname = os.path.join(out_dir, str(spkr_id) + '.csv')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True) 
    t_np = activations.detach().numpy()  
    df = pd.DataFrame(t_np) 
    df.to_csv(out_fname, index=False)
