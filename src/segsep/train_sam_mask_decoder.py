import torch
from sam_sep import SamWrapper
from loss_acc import LOGL2loss_freq
from datasets import FSD50K_Dataset

model = SamWrapper(n_fft = 2048,
                   win_length=2047,
                   spec_dim=(1024,1024),
                   sample_rate=44100,
                   resample_rate=22050)

lossfn = LOGL2loss_freq()
learning_rate = 1e-4
torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0)