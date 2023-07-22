import torch

# --------------------------------------------------------------------------------------------------
def spectral_centroid_spec(spec):
  # The frequency values corresponding to each row of the spectrogram
  freqs = torch.arange(spec.size(0))
  
  # Calculate the spectral centroid
  centroid = torch.sum(freqs * spec, axis=0) / torch.sum(spec, axis=0)
  return centroid

# --------------------------------------------------------------------------------------------------
def spectral_centroid_waveform(waveform, sample_rate=44100, n_fft=1024, hop_length=256):
    stft_result = torch.stft(waveform, 
                             n_fft=n_fft, 
                             hop_length=hop_length, 
                             window=torch.hann_window(n_fft).to(waveform.device), 
                             return_complex=True)
    
    mag_spectrum = torch.abs(stft_result)
    freqs = torch.linspace(0, sample_rate//2, mag_spectrum.shape[1])
    spectral_centroid_val = torch.sum(mag_spectrum * freqs) / torch.sum(mag_spectrum)
    return spectral_centroid_val.item()

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
  print(f"{title:20} - {str(tensor_type):15} - shape: {str(shape):29} (magnitude) max: {max:8.4f}, min {min:8.4f}, mean {mean:8.4f}, std: {std:8.4f}")