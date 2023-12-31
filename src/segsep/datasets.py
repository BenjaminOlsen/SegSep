import os
import json
import torch
import random
import torchaudio
from tqdm.auto import tqdm
from pathlib import Path

from segsep.utils import spectral_metadata_waveform

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
  
  for idx, audio_file in enumerate(tqdm(audio_files)):
    waveform, sample_rate = torchaudio.load(os.path.join(audio_dir, audio_file))
  
    spectral_centroid, spectral_bandwidth, spectral_flatness, spectral_contrast = spectral_metadata_waveform(waveform, sample_rate=sample_rate, frame_size=1024, hop_length=256)

    if verbose:
        print(f"generating audio metadata {idx}/{len(audio_files)} (max|min|mean|std):\
                spectral centroid {spectral_centroid['max']:.4f}|{spectral_centroid['min']:.4f}|{spectral_centroid['mean']:.4f}|{spectral_centroid['std']:.4f}, \
                spec bandwidth: {spectral_bandwidth['max']:.4f}|{spectral_bandwidth['min']:.4f}|{spectral_bandwidth['mean']:.4f}|{spectral_bandwidth['std']:.4f}, \
                contrast: {spectral_contrast['max']:.4f}|{spectral_contrast['min']:.4f}|{spectral_contrast['mean']:.4f}|{spectral_contrast['std']:.4f}, \
                flatness: {spectral_flatness['max']:.4f}|{spectral_flatness['min']:.4f}|{spectral_flatness['mean']:.4f}|{spectral_flatness['std']:.4f}, \
                sample_cnt {waveform.shape[1]}, sr: {sample_rate}")
    metadata.append({
        'filename': audio_file,
        'sample_cnt': waveform.shape[1],
        'sample_rate': sample_rate,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_flatness': spectral_flatness,
        'spectral_contrast': spectral_contrast
    })
  
  metadata.sort(key=lambda x: x['spectral_flatness']['mean'])
  
  with open(output_file, 'w') as f:
    json.dump(metadata, f)
  print("done")

# --------------------------------------------------------------------------------------------------
class AudioPairDataset(torch.utils.data.Dataset):
  """
  this class defines a dataset which makes random mixtures of a given set of audio defined
  in a json file given in the constructor argument, according to certain conditions on the
  certain characteristics of the audio.

  Those characteristics are:
  1) a minimimum duration
  2) minimum difference in mean spectral centroid
  3) minimum difference in mean spectral bandwidth
  4) minimum difference in mean spectral contrast
  5) minimum difference in mean spectral flatness

  The default values are taken as the mean standard deviation of the FSD dev dataset

  if the json metadata file doesn't exist, it tries to create it using generate_audio_metadata.

  It returns a mix, and the separate sources from its __getitem__ using the index as a
  random seed.

  dummy_mode creates a mix with the longer of the audios set to 0 for a more event-detection
  task
  """
  def __init__(self, audio_dir, json_path,
               centroid_diff_hz=2000.0,
               bandwidth_diff=1350.0,
               flatness_diff=0.16,
               contrast_diff=0.07,
               min_duration_s=11.0,
               dummy_mode=False):

    if json_path == None:
      json_path = 'AudioPairDataset_metadata.json'

    if not os.path.exists(json_path):
      print(f"creating {json_path}")
      generate_audio_metadata(audio_dir=audio_dir, output_file=json_path, verbose=True)

    with open(json_path, 'r') as f:
      self.data_all = json.load(f)

    self.audio_dir = audio_dir
    self.centroid_diff_hz = centroid_diff_hz
    self.bandwidth_diff = bandwidth_diff
    self.flatness_diff = flatness_diff
    self.contrast_diff = contrast_diff
    self.min_duration_s = min_duration_s
    self.data_long = [d for d in self.data_all if d['sample_cnt'] / d['sample_rate'] > min_duration_s]
    self.dummy_mode = dummy_mode


    if len(self.data_long) < 2:
      raise ValueError(f"Not enough tracks longer than {min_duration_s} seconds")

    print(f"found {len(self.data_long)} files of length {min_duration_s}s or longer")

    torch.manual_seed(0)  # You might want to fix the seed if you want consistent shuffling.
    random.shuffle(self.data_long)
    random.shuffle(self.data_all)

    print(f"AudioPairDataset init: counting audio pairs...")
    self.length = self.count_audio_pairs()
    print(f"...done")

  ####################################
  def load_audio(self, filename):
    try:
      waveform, sample_rate = torchaudio.load(filename)
      return waveform, sample_rate, filename
    except Exception as e:
      print(f"Error loading audio file {filename}: {e}")
      return None, None

  ####################################
  def audio_pair_satisfies_condition(self, idx_long, idx_all):
    info1 = self.data_long[idx_long]
    info2 =  self.data_all[idx_all]

    fn_1 = info1['filename']
    fn_2 = info2['filename']

    # look at distinct pairs only
    if fn_1 == fn_2:
      return False

    sc_1 = info1['spectral_centroid']['mean']
    sc_2 = info2['spectral_centroid']['mean']

    flat_1 = info1['spectral_flatness']['mean']
    flat_2 = info2['spectral_flatness']['mean']

    bw_1 = info1['spectral_bandwidth']['mean']
    bw_2 = info2['spectral_bandwidth']['mean']

    cont_1 = info1['spectral_contrast']['mean']
    cont_2 = info2['spectral_contrast']['mean']

    return (abs(sc_1 - sc_2) >= self.centroid_diff_hz and
            abs(flat_1 - flat_2) >= self.flatness_diff and
            abs(bw_1 - bw_2) >= self.bandwidth_diff and
            abs(cont_1 - cont_2) >= self.contrast_diff)

  ####################################
  def count_audio_pairs(self):
    count = 0
    for i in range(len(self.data_long)):
        for j in range(i + 1, len(self.data_all)):
          if self.audio_pair_satisfies_condition(i, j):
            count += 1

    return count

  ####################################
  def get_audio_pairs(self, idx):
    torch.manual_seed(idx)
    random.seed(idx)
    random.shuffle(self.data_long)
    random.shuffle(self.data_all)

    for i in range(len(self.data_long)):
      for j in range(i+1, len(self.data_all)):

        if self.audio_pair_satisfies_condition(idx_long=i, idx_all=j):
          waveform1, sample_rate1, filename1 = self.load_audio(os.path.join(self.audio_dir, self.data_long[i]['filename']))
          waveform2, sample_rate2, filename2 = self.load_audio(os.path.join(self.audio_dir, self.data_all[j]['filename']))

          if waveform1 is None or waveform2 is None:
            continue

          info1 = self.data_long[i]
          info2 = self.data_all[j]

          return waveform1, waveform2, info1, info2
    raise ValueError("No pair found with the required spectral centroid difference")

  ####################################
  def __getitem__(self, idx):
    mix, longer_audio, padded_audio, longer_info, shorter_info, pad_offset = self.get_info(idx)
    return mix, padded_audio

  ####################################
  def get_info(self, idx):
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
    #print(f"padding audio1 {len_short} -> {len_long}: {len_long-len_short} offset {pad_offset}")
    padded_audio = torch.nn.functional.pad(shorter_audio, (pad_offset, len_long-len_short-pad_offset))

    if self.dummy_mode:
      longer_audio = torch.zeros(longer_audio.shape)
    mix = longer_audio + padded_audio

    return mix, longer_audio, padded_audio, longer_info, shorter_info, pad_offset

  ####################################
  def __len__(self):
    return self.length