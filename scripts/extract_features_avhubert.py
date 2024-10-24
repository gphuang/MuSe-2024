# source: https://colab.research.google.com/drive/1bNXkfpHiVHzXQH8WjGhzQ-fsDxolpUjD#scrollTo=-D_fa6SGMpfA
# install: https://github.com/facebookresearch/av_hubert
# source activate av_hubert

import sys
import pathlib
from pathlib import Path
import pandas as pd
import tempfile
import torch
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils

import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
from base64 import b64encode

sys.path.append('/scratch/work/huangg5/tutorials/av_hubert/avhubert')
import utils as avhubert_utils
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# tools
# mkdir -p /scratch/work/huangg5/tutorials/avhubert_tools
# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O /scratch/work/huangg5/tutorials/avhubert_tools/shape_predictor_68_face_landmarks.dat.bz2
# bzip2 -d /scratch/work/huangg5/tutorials/avhubert_tools/shape_predictor_68_face_landmarks.dat.bz2
# wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O /scratch/work/huangg5/tutorials/avhubert_tools/20words_mean_face.npy

# avhubert 
model_names=['base_lrs3_iter5', ]
model_names+=['large_lrs3_iter5', 'base_vox_iter5', 'large_vox_iter5', 'base_noise_pt_noise_ft_30h', 'large_noise_pt_noise_ft_30h'] 
ffmpeg_path="/scratch/work/huangg5/.conda_envs/avhubert/bin/ffmpeg"
user_dir = "/scratch/work/huangg5/tutorials/av_hubert/avhubert/"
face_predictor_path = "/scratch/work/huangg5/tutorials/avhubert_tools/shape_predictor_68_face_landmarks.dat"
mean_face_path = "/scratch/work/huangg5/tutorials/avhubert_tools/20words_mean_face.npy"

# muse-2024
video_data = "/scratch/elec/puhe/c/muse_2024/c1_muse_perception/raw/videos/" 
feat_dir = "/scratch/elec/puhe/c/muse_2024/c1_muse_perception/feature_segments"
id_header = 'subj_id'
# /scratch/elec/puhe/c/muse_2024/c2_muse_humor/raw_data/faces/ # c2_muse_humor provides face data

def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(face_predictor_path)
  STD_SIZE = (256, 256)
  mean_face_landmarks = np.load(mean_face_path)
  stablePntsIDs = [33, 36, 39, 42, 45]
  videogen = skvideo.io.vread(input_video_path)
  frames = np.array([frame for frame in videogen])
  landmarks = []
  for frame in tqdm(frames):
      landmark = detect_landmark(frame, detector, predictor)
      landmarks.append(landmark)
  preprocessed_landmarks = landmarks_interpolate(landmarks)
  rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
  write_video_ffmpeg(rois, output_video_path, ffmpeg_path)
  return

def extract_visual_feature(video_path, ckpt_path, user_dir, is_finetune_ckpt=False):
  utils.import_user_module(Namespace(user_dir=user_dir))
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  transform = avhubert_utils.Compose([
      avhubert_utils.Normalize(0.0, 255.0),
      avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
      avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])
  frames = avhubert_utils.load_video(video_path)
  print(f"Load video {video_path}: shape {frames.shape}")
  frames = transform(frames)
  print(f"Center crop video to: {frames.shape}")
  frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
  model = models[0]
  if hasattr(models[0], 'decoder'):
    print(f"Checkpoint: fine-tuned")
    model = models[0].encoder.w2v_model
  else:
    print(f"Checkpoint: pre-trained w/o fine-tuning")
  model.to(device)
  model.eval()
  with torch.no_grad():
    # Specify output_layer if you want to extract feature of an intermediate layer
    feature, _ = model.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None)
    feature = feature.squeeze(dim=0)
  print(f"Video feature shape: {feature.shape}")
  return feature

# demo
if 0:
  #! python scripts/extract_avhubert_features.py DUMMY
  # https://github.com/facebookresearch/av_hubert/issues/36 
  model_name = 'base_lrs3_iter5'
  ckpt_path = f"/scratch/work/huangg5/tutorials/avhubert_pretrained_models/{model_name}.pt"
  # muse sample
  origin_clip_path = "/scratch/elec/puhe/c/muse_2024/c1_muse_perception/raw/videos/0.mp4"
  mouth_roi_path = "/scratch/elec/puhe/c/muse_2024/c1_muse_perception/raw/roi/0.mp4"
  # tutorial sample
  # !wget --content-disposition https://dl.fbaipublicfiles.com/avhubert/demo/avhubert_demo_video_8s.mp4 -O /scratch/work/huangg5/tutorials/data_example/clip.mp4
  origin_clip_path = "/scratch/work/huangg5/tutorials/data_example/clip.mp4"
  mouth_roi_path = "/scratch/work/huangg5/tutorials/data_example/roi.mp4"
  
  if False: preprocess_video(origin_clip_path, mouth_roi_path, face_predictor_path, mean_face_path) # extract mouth roi
  if False: pass # TODO extract face, lips, eyes, hands roi, avhubert is limited only on mouth_roi
  if True: feature = extract_visual_feature(mouth_roi_path, ckpt_path, user_dir) # extract video features from mouth_roi
  if False: feature = extract_visual_feature(origin_clip_path, ckpt_path, user_dir) # extract video features
  
  out_fname = "/scratch/work/huangg5/tutorials/data_example/avhubert-feat.csv"
  t_np = feature.detach().numpy()  
  print(t_np.shape)
  df = pd.DataFrame(t_np) 
  df.to_csv(out_fname, index=False)
  sys.exit(0)
  
# whether to regnerate feature files
overwrite = True

# iteration
onlyfiles = [f for f in os.listdir(video_data) if os.path.isfile(os.path.join(video_data, f)) and not f.startswith('.')]
assert len(onlyfiles)==177
for model_name in model_names:
  ckpt_path = f"/scratch/work/huangg5/tutorials/avhubert_pretrained_models/{model_name}.pt"
  for _file in tqdm(onlyfiles):
    f_id = Path(_file).stem
    feat_name = 'avhubert-' + ('-').join(model_name.split('_'))
    print(f'Spkr: {f_id}, feat: {feat_name}')
    origin_clip_path = os.path.join(video_data, str(f_id) + '.mp4')
    out_dir = os.path.join(feat_dir, feat_name) # save features
    out_fname = os.path.join(out_dir, str(f_id) + '.csv')
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    if overwrite or not os.path.exists(out_fname):
      feature = extract_visual_feature(origin_clip_path, ckpt_path, user_dir) # extract video features
      t_np = feature.detach().numpy()
      columns = ['feat_' + str(i)  for i in range(t_np.shape[1])]
      df = pd.DataFrame(t_np, columns=columns) 
      # add header column. !Perception data_parser did not throw exception!
      df['timestamp'] = [str(i*500)  for i in range(t_np.shape[0])]
      df[id_header] = f_id
      df = df[['timestamp', id_header] + [ col for col in df.columns if col not in ['timestamp', id_header] ] ]
      df.to_csv(out_fname, index=False)
      #df=pd.read_csv(out_fname, index_col=0)
      #print(df.shape, df.head(3), out_fname)
      #sys.exit(0)
    