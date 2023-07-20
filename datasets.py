import torch
import torchaudio
from pathlib import Path

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
