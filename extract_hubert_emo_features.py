# extract emotional features from audio using HuBert
# source: https://huggingface.co/superb/hubert-large-superb-er

import sys
import torch
import librosa
import datasets
from datasets import load_dataset
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

"""# demo
def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example

# load a demo dataset and read audio files
dataset = load_dataset("anton-l/superb_demo", "er", split="session1")
dataset = dataset.map(map_to_array)

# compute attention masks and normalize the waveform if needed
inputs = feature_extractor(dataset[:4]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
print(inputs)

logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]

print(logits.shape, predicted_ids.shape)"""
# sys.exit(0)

# muse
audio_data = '/scratch/elec/puhe/c/muse_2024/c1_muse_perception/raw/wav'
dataset = load_dataset("audiofolder", data_dir=audio_data, drop_labels=True, drop_metadata=True)
dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
audio_arrays = [x["array"] for x in dataset['train']["audio"]]
inputs = feature_extractor(audio_arrays[:4], sampling_rate=16000, padding=True, return_tensors="pt")

logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
print(logits.shape, predicted_ids.shape)

# iterate?

# save feature for each audio file hubert-superb-er