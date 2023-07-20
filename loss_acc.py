import torch

# --------------------------------------------------------------------------------------------------
def LOGL2loss_freq(Y1, Y2):
  assert (Y1.shape == Y2.shape), f"Shapes do not match: {Y1.shape} vs {Y2.shape}"
  N = Y1.numel()
  return (10/(2*N))*torch.log10(torch.sum((torch.abs(Y1) - torch.abs(Y2))**2 )+ 1e-7)

# --------------------------------------------------------------------------------------------------
def si_snr(pred_audio, target_audio):
  target = target_audio.reshape(-1) # flatten
  pred = pred_audio.reshape(-1) # flatten
  eps = 1e-8
  s_target = target * (pred.dot(target) / (eps + target.dot(target)))
  e_noise = pred - s_target
  SI_SNR = 10 * torch.log10(eps + (s_target.norm() / (eps + e_noise.norm())))
  return SI_SNR