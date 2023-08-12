import torch
import numpy as np
import essentia.standard as es

# --------------------------------------------------------------------------------------------------
def spectral_centroid_spec(spec):
  # The frequency values corresponding to each row of the spectrogram
  freqs = torch.arange(spec.size(0))
  
  # Calculate the spectral centroid
  centroid = torch.sum(freqs * spec, axis=0) / torch.sum(spec, axis=0)
  return centroid

# --------------------------------------------------------------------------------------------------
def arr_stats(arr: np.array):
  """
  returns a dictionary containing max, min, mean, std of the array
  """
  return {
    "max": np.max(arr),
    "min": np.min(arr),
    "mean": np.mean(arr),
    "std": np.std(arr)
    }

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

  # the torch spectrogram shape is [Freq, Time]
  freq_indices = torch.linspace(0, sample_rate//2, mag_spectrum.shape[0]).unsqueeze(1)
  time_indices = torch.arange(mag_spectrum.shape[1]).float()

  # calculate time-centroid
  # Sum along the frequency axis
  ### only sum where the mag_spectrum is not zero!
  intensity_sum = torch.sum(mag_spectrum, dim=0) 
  non_zero_indices = torch.nonzero(intensity_sum, as_tuple=True)[0]
  non_zero_intensities = intensity_sum[non_zero_indices]
  non_zero_time_indices = time_indices[non_zero_indices]

  temporal_centroid_index = torch.sum(non_zero_intensities * non_zero_time_indices) / torch.sum(non_zero_intensities)
  temporal_centroid_val = (temporal_centroid_index * hop_length) / sample_rate
  temporal_centroid_val = torch.nan_to_num(temporal_centroid_val, nan=0.0)
  temporal_centroid_val = temporal_centroid_val.item()
    
  # Calculate freq-centroid
  #sum the same frequency bin for all hops:
  spectral_centroid_val = torch.sum(mag_spectrum * freq_indices, dim=0) / torch.sum(mag_spectrum, dim=0)
  #nan to zero
  spectral_centroid_val = torch.nan_to_num(spectral_centroid_val, nan=0.0)
  mean_spectral_centroid = torch.mean(spectral_centroid_val).item()
  return {"time_centroid_s": temporal_centroid_val, "spectral_centroid_hz": mean_spectral_centroid}

# ------------------------------------------------------------------------------
def spectral_flatness_waveform(waveform, frame_size=1024, hop_length=256):
  """
  return the mean spectral flatness of the waveform*


  Args:
  - waveform (torch.tensor): audio
  - frame_size (int): FFT size
  - hop_length (int): FFT hop length

  Returns:
  - contrast (float): mean spectral flatness 
  """

  audio = waveform.squeeze().numpy()

  w = es.Windowing(type='hann')
  spectrum = es.Spectrum()
  flatness = es.Flatness()
  spectral_flatness_values = []

  for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_length):
    spec = spectrum(w(frame))
    flat = flatness(spec)
    spectral_flatness_values.append(flat)
  
  return np.mean(spectral_flatness_values)

# ------------------------------------------------------------------------------
def spectral_contrast_waveform(waveform, sample_rate=44100, frame_size=1024, hop_length=256):
  """
  returns the mean spectral contrast of the waveform.

  Args:
  - waveform (torch.tensor): audio
  - sample_rate (int): sample rate in Hz
  - frame_size (int): FFT size
  - hop_length (int): FFT hop length

  Returns:
  - contrast (float): mean spectral contrast 
  """
  audio = waveform.squeeze().numpy()

  w = es.Windowing(type='hann')
  spectrum = es.Spectrum()
  contrast = es.SpectralContrast(frameSize=frame_size, sampleRate=sample_rate)

  spectral_contrasts = []

  for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_length):
    spec = spectrum(w(frame))
    sc = contrast(spec)
    spectral_contrasts.append(sc)
  
  return np.mean(spectral_contrasts)

# ------------------------------------------------------------------------------
def spectral_bandwidth_waveform(waveform, sample_rate=44100, frame_size=1024, hop_length=256):
  """
  Computes the mean spectral bandwidth of a waveform

  Args:
  - waveform (torch.tensor): audio
  - sample_rate (int): sample rate in Hz
  - frame_size (int): FFT size
  - hop_length (int): FFT hop length

  Returns:
  - bandwidth (float): mean spectral bandwidth in Hz
  """
  audio = waveform.squeeze().numpy()
  w = es.Windowing(type='hann')
  spectrum = es.Spectrum()

  spectral_bandwidths = []

  for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_length):
    spec = spectrum(w(frame))
    num_bins = spec.size # Number of frequency bins
    freqs = np.linspace(0, sample_rate // 2, num_bins)
    sum_spec = np.sum(spec)
    if np.isclose(sum_spec, 0):
      bandwidth = 0.0
    else:
      spectral_centroid = np.sum(freqs * spec) / sum_spec
      bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spec) / np.sum(spec))

    if np.isnan(bandwidth):
      bandwidth = 0.0

    spectral_bandwidths.append(bandwidth)
  
  return np.mean(spectral_bandwidths)

# --------------------------------------------------------------------------------------------------
def spectral_metadata_waveform(waveform, sample_rate=44100, frame_size=1024, hop_length=256):
  """
  Computes the spectral centroid, bandwidth, flatness, and contrast of a given waveform

  Args:
  - waveform (torch.tensor): audio
  - sample_rate (int): sample rate in Hz
  - frame_size (int): FFT size
  - hop_length (int): FFT hop length

  Returns:
  tuple containing (centroid, bandwidth, flatness, contrast)
  - spectral centroid (float): spectral centroid in Hz
  - bandwidth (float): mean spectral bandwidth in Hz
  - flatness (float): mean spectral flatness
  - contrast (float): mean spectral contrast 
  """
  audio = waveform.squeeze().numpy()
  w = es.Windowing(type='hann')
  spectrum = es.Spectrum()

  num_frames = 1 + (len(audio) - frame_size) // hop_length
  
  spectral_centroids = np.zeros(num_frames)
  spectral_bandwidths = np.zeros(num_frames)

  flatness = es.Flatness()
  spectral_flatness_values = np.zeros(num_frames)

  contrast = es.SpectralContrast(frameSize=frame_size, sampleRate=sample_rate)
  spectral_contrasts = np.zeros(num_frames)

  cur_frame = 0
  for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_length):
    if cur_frame >= len(spectral_flatness_values):
      break

    spec = spectrum(w(frame))
    
    #######################################
    # spectral flatness
    flat = flatness(spec)
    spectral_flatness_values[cur_frame] = flat

    #######################################
    # spectral contrast
    sc = np.mean(contrast(spec)[0]) # outputs two vectors spectralContrast, spectralValley
    spectral_contrasts[cur_frame] = sc

    ########################################
    # spectral bandwidth & centroid
    num_bins = spec.size # Number of frequency bins
    freqs = np.linspace(0, sample_rate // 2, num_bins)
    sum_spec = np.sum(spec)
    if np.isclose(sum_spec, 0):
      bandwidth = 0.0
      spectral_centroid = 0.0
    else:
      spectral_centroid = np.sum(freqs * spec) / sum_spec
      bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spec) / np.sum(spec))

    if np.isnan(bandwidth):
      bandwidth = 0.0

    spectral_centroids[cur_frame] = spectral_centroid
    spectral_bandwidths[cur_frame] = bandwidth
    
    cur_frame += 1
  
  flat_data = arr_stats(spectral_flatness_values)
  contrast_data = arr_stats(spectral_contrasts)
  bw_data = arr_stats(spectral_bandwidths)
  centroid_data = arr_stats(spectral_centroids)

  return centroid_data, bw_data, flat_data, contrast_data


# --------------------------------------------------------------------------------------------------
def calculate_energy(audio):
  return torch.sum(audio**2)

# --------------------------------------------------------------------------------------------------
def should_skip_chunk(audio, threshold=1e-3, exact_size=None):
  should_skip = True

  if exact_size and audio.shape[1] == exact_size:
    should_skip = False
  
  if calculate_energy(audio) > threshold:
    should_skip = False

  return should_skip

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