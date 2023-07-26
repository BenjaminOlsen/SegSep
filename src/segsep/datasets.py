import os
import json
import torch
import random
import torchaudio
from pathlib import Path

from segsep.utils import spectral_centroid_waveform

# --------------------------------------------------------------------------------------------------
class FSD50K_Dataset(torch.utils.data.Dataset):
  def __init__(self, fsd_path):
    super(FSD50K_Dataset, self).__init__()
    self.fsd_path = fsd_path
    self.paths = list(Path(self.fsd_path).glob("*.wav"))
    
  def __len__(self) -> int:
    return len(self.paths)

  def load_audio(self, idx):
    waveform, sample_rate = torchaudio.load(self.paths[idx])
    return waveform, sample_rate

  def __getitem__(self, idx: int):
    waveform, sr = self.load_audio(idx)
    return waveform, sr

# --------------------------------------------------------------------------------------------------
class MusdbDataset(torch.utils.data.Dataset):
  def __init__(self, musdb_data):
    super(MusdbDataset, self).__init__()
    self.musdb = musdb_data

  def __len__(self) -> int:
    return len(self.musdb)

  def __getitem__(self, index: int):
    track = self.musdb[index]
    #track.stems[0].shape # [channels, sample_cnt]
    mix_audio = torch.Tensor(track.stems[0].T)
    vocal_audio = torch.Tensor(track.stems[4].T)
    return mix_audio, vocal_audio

# --------------------------------------------------------------------------------------------------
def generate_audio_metadata(audio_dir, output_file, verbose=False):
  audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') or f.endswith('.mp3')]
  
  metadata = []
  
  for idx, audio_file in enumerate(audio_files):
    waveform, sample_rate = torchaudio.load(os.path.join(audio_dir, audio_file))
    spectral_centroid_data = spectral_centroid_waveform(waveform, sample_rate)
    sc_freq = spectral_centroid_data["spectral_centroid_hz"]
    sc_time = spectral_centroid_data["time_centroid_s"]
    if verbose:
        print(f"generating audo metadata {idx}/{len(audio_files)} : spectral centroid {sc_freq:.4f}, time centroid: {sc_time}, sample_cnt {waveform.shape[1]}, sr: {sample_rate}")
    metadata.append({
        'filename': audio_file,
        'sample_cnt': waveform.shape[1],
        'sample_rate': sample_rate,
        'spectral_centroid_hz': sc_freq,
        'time_centroid_s': sc_time
    })
  
  metadata.sort(key=lambda x: x['spectral_centroid_hz'])
  
  with open(output_file, 'w') as f:
    json.dump(metadata, f)
  print("done")

# --------------------------------------------------------------------------------------------------
class AudioPairDataset(torch.utils.data.Dataset):
  """
  this class defines a dataset which makes random mixtures of a given set of audio defined 
  in a json file given in the constructor argument, according to certain conditions on the
  spectral centroid difference and a minimum duration.

  It returns a mix, and the separate sources from its __getitem__ using the index as a 
  random seed.

  dummy_mode creates a mix with the longer of the audios set to 0 for a more event-detection
  task
  """
  def __init__(self, audio_dir, json_path, centroid_diff_hz=2000.0, min_duration_s=11.0, dummy_mode=False):
    with open(json_path, 'r') as f:
      self.data_all = json.load(f)
    self.audio_dir = audio_dir
    self.centroid_diff_hz = centroid_diff_hz
    self.min_duration_s = min_duration_s
    self.data_long = [d for d in self.data_all if d['sample_cnt'] / d['sample_rate'] > min_duration_s]
    self.dummy_mode = dummy_mode

    if len(self.data_long) < 2:
      raise ValueError(f"Not enough tracks longer than {min_duration_s} seconds")

    print(f"found {len(self.data_long)} files of length {min_duration_s}s or longer")

  def load_audio(self, filename):
    try:
      waveform, sample_rate = torchaudio.load(filename)
      return waveform, sample_rate, filename
    except Exception as e:
      print(f"Error loading audio file {filename}: {e}")
      return None, None
  
  def get_audio_pairs(self, idx):
    torch.manual_seed(idx)
    random.shuffle(self.data_long)
    random.shuffle(self.data_all)

    for i in range(len(self.data_long)):
      for j in range(i+1, len(self.data_all)):
        sc_1 = self.data_long[i]['spectral_centroid_hz']
        sc_2 = self.data_all[j]['spectral_centroid_hz']
        tc_1 = self.data_long[i]['time_centroid_s']
        tc_2 = self.data_all[i]['time_centroid_s']
        centroid_diff_hz_ij = abs(sc_1 - sc_2)
        if centroid_diff_hz_ij > self.centroid_diff_hz:
          waveform1, sample_rate1, filename1 = self.load_audio(os.path.join(self.audio_dir, self.data_long[i]['filename']))
          waveform2, sample_rate2, filename2 = self.load_audio(os.path.join(self.audio_dir, self.data_all[j]['filename']))
          if waveform1 is None or waveform2 is None:
            continue

          info1 = {"sample_rate" : sample_rate1,
                  "filename": filename1,
                  "spectral_centroid": sc_1,
                  "time_centroid": tc_1}
          
          info2 = {"sample_rate" : sample_rate2,
                  "filename": filename2,
                  "spectral_centroid": sc_2,
                  "time_centroid": tc_2}
          
          return waveform1, waveform2, info1, info2
    raise ValueError("No pair found with the required spectral centroid difference")

  def __getitem__(self, idx):
    x1, x2, info1, info2 = self.get_audio_pairs(idx)

    len1 = x1.shape[1]
    len2 = x2.shape[1]

    if len1 < len2:
      shorter_audio = x1
      longer_audio = x2
      longer_info = info2
      shorter_info= info1
    else:
      shorter_audio = x2
      longer_audio = x1
      longer_info = info1
      shorter_info = info2

    len_short = shorter_audio.shape[1]
    len_long = longer_audio.shape[1]

    # put the shorter audio at a random starting location within the longer
    pad_offset = random.randint(0, len_long-len_short)
    print(f"padding audio1 {len_short} -> {len_long}: {len_long-len_short} offset {pad_offset}")
    padded_audio = torch.nn.functional.pad(shorter_audio, (pad_offset, len_long-len_short-pad_offset))
    
    if self.dummy_mode:
      longer_audio = torch.zeros(longer_audio.shape)
    mix = longer_audio + padded_audio

    return mix, longer_audio, padded_audio, longer_info, shorter_info, pad_offset
    

  def __len__(self):
    return len(self.data_long)