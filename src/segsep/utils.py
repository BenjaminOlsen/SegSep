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
  """
  calculates the time-freq centroid of a waveform, essentially as a 'center of mass',
  returns:
  {"time_centroid"  time of centroid (seconds),
   "spectral_centroid": spectral centroid (Hz))
  }
  """
  stft_result = torch.stft(waveform, 
                            n_fft=n_fft, 
                            hop_length=hop_length, 
                            window=torch.hann_window(n_fft).to(waveform.device), 
                            center=True,
                            onesided=True,
                            return_complex=True)

  mag_spectrum = torch.abs(stft_result).squeeze()

  freq_indices = torch.linspace(0, sample_rate//2, mag_spectrum.shape[0]).unsqueeze(1)
  time_indices = torch.arange(mag_spectrum.shape[1]).float()

  # calculate time-centroid
  # Sum along the frequency axis
  intensity_sum = torch.sum(mag_spectrum, dim=0) # Assuming the spectrogram shape is [Freq, Time]
  temporal_centroid_index = torch.sum(intensity_sum * time_indices) / torch.sum(intensity_sum)
  temporal_centroid_val = (temporal_centroid_index * hop_length) / sample_rate # convert time bin index to seconds
  temporal_centroid_val = temporal_centroid_val.item()
    
  # Calculate freq-centroid
  #sum the same frequency bin for all hops:
  spectral_centroid_val = torch.sum(mag_spectrum * freq_indices, dim=0) / torch.sum(mag_spectrum, dim=0)
  #nan to zero
  spectral_centroid_val = torch.nan_to_num(spectral_centroid_val, nan=0.0)
  mean_spectral_centroid = torch.mean(spectral_centroid_val).item()
  return {"time_centroid_s": temporal_centroid_val, "spectral_centroid_hz": mean_spectral_centroid}

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