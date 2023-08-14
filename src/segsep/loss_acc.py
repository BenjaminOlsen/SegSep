import torch


"""
N.B. The source for these loss functions is the paper "ON LOSS FUNCTIONS AND EVALUATION METRICS FOR MUSIC SOURCE SEPARATION"
by Enric Guso, Jordi Pons, Santiago Pascual, Joan Serra.

In the formulae, the common multiplicative factor is given in the paper as 1/(N*Omega*K) where:
N is the number of frames (time dimension) of a spectrogram; Omega is the number of frequency bins (frequency dimension), 
and K is the number of sources. Here, the size of a tensor Y is N*Omega = Y.numel(); and K is always 2.
"""

# --------------------------------------------------------------------------------------------------
def LOGL2loss_freq(Y1, Y2):
  assert (Y1.shape == Y2.shape), f"Shapes do not match: {Y1.shape} vs {Y2.shape}"
  N = Y1.numel()
  return (10/(2*N))*torch.log10(torch.sum((torch.abs(Y1) - torch.abs(Y2))**2 )+ 1e-7)

# --------------------------------------------------------------------------------------------------
def LOGL1loss_freq(Y1, Y2):
  assert (Y1.shape == Y2.shape), f"Shapes do not match: {Y1.shape} vs {Y2.shape}"
  N = Y1.numel()
  return (10/(2*N))*torch.log10(torch.sum(torch.abs(Y1) - torch.abs(Y2))+ 1e-7)

# --------------------------------------------------------------------------------------------------
def L1loss_freq(Y1, Y2):
  assert (Y1.shape == Y2.shape), f"Shapes do not match: {Y1.shape} vs {Y2.shape}"
  N = Y1.numel()
  return (1/2*N)*torch.sum(torch.abs(Y1) - torch.abs(Y2))

# --------------------------------------------------------------------------------------------------
def L2loss_freq(Y1, Y2):
  assert (Y1.shape == Y2.shape), f"Shapes do not match: {Y1.shape} vs {Y2.shape}"
  N = Y1.numel()
  return (1/2*N)*torch.sum((torch.abs(Y1) - torch.abs(Y2))**2)

# --------------------------------------------------------------------------------------------------
def si_snr(pred_audio, target_audio):
  target = target_audio.reshape(-1) # flatten
  pred = pred_audio.reshape(-1) # flatten
  eps = 1e-8
  s_target = target * (pred.dot(target) / (eps + target.dot(target)))
  e_noise = pred - s_target
  SI_SNR = 10 * torch.log10(eps + (s_target.norm() / (eps + e_noise.norm())))
  return SI_SNR