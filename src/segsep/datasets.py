import os
import json
import torch
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
def generate_audio_metadata(audio_dir, output_file):
    # Get list of audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') or f.endswith('.mp3')]
    
    # List to store metadata
    metadata = []
    
    for audio_file in audio_files:
        waveform, sample_rate = torchaudio.load(os.path.join(audio_dir, audio_file))
        sc = spectral_centroid_waveform(waveform, sample_rate)
        
        metadata.append({
            'filename': audio_file,
            'sample_cnt': waveform.shape[1],
            'sample_rate': sample_rate,
            'spectral_centroid': sc
        })
    
    metadata.sort(key=lambda x: x['spectral_centroid'])
    
    with open(output_file, 'w') as f:
        json.dump(metadata, f)