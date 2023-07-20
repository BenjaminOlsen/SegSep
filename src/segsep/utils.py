import torch

# --------------------------------------------------------------------------------------------------
def spectral_centroid(spec):
  # The frequency values corresponding to each row of the spectrogram
  freqs = torch.arange(spec.size(0))
  
  # Calculate the spectral centroid
  centroid = torch.sum(freqs * spec, axis=0) / torch.sum(spec, axis=0)
  return centroid

# --------------------------------------------------------------------------------------------------
def calculate_energy(audio):
  return torch.sum(audio**2)

# --------------------------------------------------------------------------------------------------
def should_skip_chunk(audio, threshold=1e-3):
  return calculate_energy(audio) < threshold

# --------------------------------------------------------------------------------------------------
def print_tensor_stats(t, title=None):
  tensor_type = t.dtype
  shape = t.shape
  t = torch.abs(t)
  max = torch.max(t)
  min = torch.min(t)
  mean = torch.mean(t)
  std = torch.std(t)
  print(f"{title:40} - {str(tensor_type):15} - shape: {str(shape):29} (magnitude) max: {max:8.4f}, min {min:8.4f}, mean {mean:8.4f}, std: {std:8.4f}")