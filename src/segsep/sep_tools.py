import os
import sys
#import musdb
import torch
import random
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchaudio.transforms as T

import segsep.utils

from pathlib import Path
from random import randint
from tqdm.auto import tqdm
from statistics import mean
from datetime import datetime


# --------------------------------------------------------------------------------------------------  
def train(model, dataloader, optimizer, loss_fn, acc_fn, device):
  model.train()
  epoch_loss = 0
  epoch_acc = 0

  # automatic mixed precision scaler:
  scaler = torch.cuda.amp.GradScaler()

  for idx, (mix_audio, vocal_audio) in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()

    # Assumes that the dataloader returns tensors of size [num_samples]
    # and the audio length is larger than spec_len_in_s * sample_rate
    # Divide the audio into chunks of spec_len_in_s seconds
    spec_len_in_s = 5.0
    sample_rate = 44100

    # choose the hop_length so that the resulting spectrogram is precisely spec_dim[0] frames
    # -> make chunk_size a clean multiple of (2*spec_dim[0]) samples around spec_len_in_s seconds length
    chunk_cnt = int(spec_len_in_s * sample_rate / (2 * model.spec_dim[0]))
    chunk_size = 2 * model.spec_dim[0] * chunk_cnt
    real_spec_len_s = chunk_size / sample_rate
    #print(f"target chunk len in s: {spec_len_in_s} -> rounded to {chunk_cnt} frames of {2*model.spec_dim[0]} samples -> {real_spec_len_s}s")

    # add a 3rd channel to each audio tensor:
    vocal_audio_ch3=(vocal_audio[:,0]-vocal_audio[:,1]).unsqueeze(0)
    mix_audio_ch3=(mix_audio[:,0]-mix_audio[:,1]).unsqueeze(0)

    mix_audio = torch.cat((mix_audio, mix_audio_ch3), dim=1)
    vocal_audio = torch.cat((vocal_audio, vocal_audio_ch3), dim=1)

    track_loss = 0
    track_acc = 0
    chunk_cnt = 0
    skip_chunk_cnt = 0
    sample_cnt = mix_audio.shape[2] # [batch, channel, audio]

    mix_audio = mix_audio.squeeze().to(device)
    vocal_audio = vocal_audio.squeeze().to(device)
    for start_idx in range(0, sample_cnt, chunk_size):
      end_idx = start_idx + chunk_size
      mix_chunk = mix_audio[:, start_idx:end_idx]
      vocal_chunk = vocal_audio[:, start_idx:end_idx]
      #print(f"doing idx {start_idx}:{end_idx} - {float(end_idx)/mix_audio_3.shape[1]:.3f}")

      if utils.should_skip_chunk(mix_chunk) or utils.should_skip_chunk(vocal_chunk):
        skip_chunk_cnt += 1
        continue

      #print(f"mix chunk shape {mix_chunk.shape}")
      if torch.isnan(mix_chunk).any():
        print("input data contains nan!")
        mix_chunk = torch.nan_to_num(mix_chunk)

      # context for automatic mixed precision
      with torch.cuda.amp.autocast():
        pred_audio, mix_mag, mix_phase, pred_spec, pred_mask = model(mix_chunk)

        if torch.isnan(pred_audio).any():
          print("prediction contains nan!")
          pred_audio = torch.nan_to_num(pred_audio)
        trim_idx = min(pred_audio.shape[1], vocal_chunk.shape[1])

        # calculate loss/acc on SPEC
        vocal_spec, vocal_phase = model.encoder(vocal_chunk)
        loss = loss_fn(pred_spec.squeeze(), vocal_spec)
        acc = acc_fn(pred_spec, vocal_spec)

      epoch_loss += loss.item()
      track_loss += loss.item()

      epoch_acc += acc.item()
      track_acc += acc.item()
      chunk_cnt += 1

      # traditional backward pass
      #loss.backward()
      #optimizer.step()

      # backward pass with automatic mixed precision
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      # madre mia
      del pred_audio
      del loss
      del acc

    del mix_audio
    del vocal_audio
    torch.cuda.empty_cache()
    print(f"TRAIN track {idx}/{len(dataloader)} loss: {track_loss}, track acc: {track_acc:.4f}, skip chunk cnt: {skip_chunk_cnt}/{chunk_cnt}")
  return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

# --------------------------------------------------------------------------------------------------
def validate(model, dataloader, loss_fn, acc_fn, device):
  model.eval()
  epoch_loss = 0
  epoch_acc = 0

  with torch.inference_mode():

    for idx, (mix_audio, vocal_audio) in enumerate(tqdm(dataloader)):
      # Assuming that the dataloader returns tensors of size [num_samples]
      # and the audio length is larger than spec_len_in_s * sample_rate
      # We divide the audio into chunks of spec_len_in_s seconds
      spec_len_in_s = 5.0
      sample_rate = 44100
      # choose the hop_length so that the resulting spectrogram is precisely spec_dim[0] frames
      # -> make chunk_size a clean multiple of (2*spec_dim[0]) samples around spec_len_in_s seconds length
      chunk_cnt = int(spec_len_in_s * sample_rate / (2 * model.spec_dim[0]))
      chunk_size = 2 * model.spec_dim[0] * chunk_cnt
      real_spec_len_s = chunk_size / sample_rate
      #print(f"target chunk len in s: {spec_len_in_s} -> rounded to {chunk_cnt} frames of {2*model.spec_dim[0]} samples -> {real_spec_len_s}s")

      # add a 3rd channel to each audio tensor:
      vocal_audio_ch3=(vocal_audio[:,0]-vocal_audio[:,1]).unsqueeze(0)
      mix_audio_ch3=(mix_audio[:,0]-mix_audio[:,1]).unsqueeze(0)

      mix_audio = torch.cat((mix_audio, mix_audio_ch3), dim=1)
      vocal_audio = torch.cat((vocal_audio, vocal_audio_ch3), dim=1)

      track_loss = 0
      track_acc = 0
      chunk_cnt = 0
      skip_chunk_cnt = 0
      sample_cnt = mix_audio.shape[2] # [batch, channel, audio]

      mix_audio = mix_audio.squeeze().to(device)
      vocal_audio = vocal_audio.squeeze().to(device)
      for start_idx in range(0, sample_cnt, chunk_size):
        end_idx = start_idx + chunk_size
        mix_chunk = mix_audio[:, start_idx:end_idx]
        vocal_chunk = vocal_audio[:, start_idx:end_idx]
        #print(f"doing idx {start_idx}:{end_idx} - {float(end_idx)/mix_audio.shape[1]:.3f}")

        if utils.should_skip_chunk(mix_chunk) or utils.should_skip_chunk(vocal_chunk):
          skip_chunk_cnt += 1
          continue

        #print(f"mix chunk shape {mix_chunk.shape}")
        if torch.isnan(mix_chunk).any():
          print("input data contains nan!")
          mix_chunk = torch.nan_to_num(mix_chunk)
        pred_audio, mix_mag, mix_phase, pred_spec, pred_mask = model(mix_chunk)
        trim_idx = min(pred_audio.shape[1], vocal_chunk.shape[1])

        # calculate loss/acc on SPEC
        vocal_spec, vocal_phase = model.encoder(vocal_chunk)
        loss = loss_fn(pred_spec.squeeze(), vocal_spec)
        acc = acc_fn(pred_spec, vocal_spec)

        epoch_loss += loss.item()
        track_loss += loss.item()

        epoch_acc += acc.item()
        track_acc += acc.item()

        chunk_cnt += 1

        # madre mia
        del pred_audio
        del loss
      del mix_audio
      del vocal_audio
      torch.cuda.empty_cache()
      print(f"TEST track {idx}/{len(dataloader)} loss: {track_loss:.8f}, track acc: {track_acc:.4f}, skip chunk cnt: {skip_chunk_cnt}/{chunk_cnt}")
      torch.cuda.empty_cache()
  return epoch_loss / len(dataloader), epoch_acc / len(dataloader)