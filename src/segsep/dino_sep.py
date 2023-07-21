import torch
import torch.nn.functional as F
import torchaudio.transforms as T

# --------------------------------------------------------------------------------------------------
class FeatureTransformer(torch.nn.Module):
  def __init__(self, in_channels, num_channels=512, tokenW=16, tokenH=16):
    super().__init__()
    self.in_channels = in_channels
    self.num_channels = num_channels
    self.tokenW = tokenW
    self.tokenH = tokenH
    self.transformerlayer = torch.nn.TransformerEncoderLayer(d_model=in_channels, nhead=8)
    self.transformer = torch.nn.TransformerEncoder(self.transformerlayer, num_layers=1)
    self.conv = torch.nn.Conv1d(in_channels, num_channels, kernel_size=1)

  def forward(self, x):
    # x shape: in_channels, height, width
    print(f"feature transformer input shape: {x.shape} in_channels, height, width")
    x = x.view(self.in_channels, -1)  # flatten the height and width into the sequence dimension
    print(f"feature transformerx.shape after flatten the height and width into the sequence dimension {x.shape}")
    x = x.transpose(0, 1)  # swap the sequence and channels dimensions
    print(f"feature transformerx.shape after swapping sequence/channel dimensions {x.shape} num_tokens, in_channels")
    # x shape: num_tokens, in_channels
    x = self.transformer(x.unsqueeze(1))  # transformer on the token dimension, add a batch dimension
    print(f"feature transformerx.shape after transformer pass on the token dimension and adding batch dim{ x.shape} num_tokens, batch idx in_channels")
    x = x.transpose(0, 1)  # swap back the sequence and channels dimensions
    print(f"feature transformerx.shape back to batch, in_channels, num_tokens: {x.shape}")
    x = self.conv(x)  # conv1d on the token dimension
    print(f"feature transformerx.shape after convolution along the token dimension: {x.shape}")
    # x shape: num_channels, num_tokens
    #x = x.view(self.num_channels, self.tokenW, self.tokenH)  # reshape back to 3D
    x = x.view(-1, self.num_channels, self.tokenH, self.tokenW)
    # x shape: num_channels, height, width
    print(f"feature transformerx.shape back to 3D view {x.shape} [num channels, feature map width, height]")
    return x
  
# --------------------------------------------------------------------------------------------------
class DinoSeg(torch.nn.Module):
  def __init__(self, n_fft=2048,
               win_length=2047,
               spec_dim=(1024, 1024),
               sample_rate=44100,
               resample_rate=22050,
               num_class=1) -> None:
    super().__init__()
    n=512
    self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    self.n_fft = n_fft
    self.win_length = win_length
    self.spec_dim = spec_dim
    self.n_fft = 2*(self.spec_dim[1]-1)
    self.win_length = self.n_fft
    self.hop_length = self.n_fft//8
    self.sample_rate = sample_rate
    self.resample_rate = resample_rate
    self.downsampler = T.Resample(orig_freq=self.sample_rate, new_freq=self.resample_rate)
    self.upsampler = T.Resample(orig_freq=self.resample_rate, new_freq=self.sample_rate)
    self.input_chunk_size = (self.sample_rate / self.resample_rate) * (self.spec_dim[0]-1) * self.hop_length
    
    if not self.input_chunk_size.is_integer():
      print(f"WARNING, noninteger input chunk size, choose your sample rate, resample rate to divide cleanly!")
    print(f"SamWrapper model n_fft: {self.n_fft}, win len: {self.win_length}, hop len: {self.hop_length}, sample/resample: {self.sample_rate / self.resample_rate} -> input_chunk_size {self.input_chunk_size}")
    
    self.input_chunk_size = int(self.input_chunk_size)
    self.classlayer_448 = FeatureTransformer(in_channels=1024,num_channels=n,tokenW=32,tokenH=32)
    self.classlayer_224 = FeatureTransformer(in_channels=1024,num_channels=n,tokenW=16,tokenH=16)
    self.selu = torch.nn.SELU()
    self.to_448 = torch.nn.Sequential(
      torch.nn.Conv2d(n,n,kernel_size=7,stride=1,padding=1,bias=False),
      torch.nn.Upsample(scale_factor=2),
      torch.nn.Conv2d(n,n//2,kernel_size=3,stride=1,padding=1,bias=False),
      torch.nn.BatchNorm2d(n//2),
      torch.nn.ReLU(inplace=True),
      torch.nn.Upsample(scale_factor=2),
      torch.nn.Conv2d(n//2,n//4,kernel_size=3,stride=1,padding=1,bias=False),
      torch.nn.BatchNorm2d(n//4),
      torch.nn.ReLU(inplace=True),
      torch.nn.Upsample(scale_factor=2),
      torch.nn.Conv2d(n//4,n//8,kernel_size=3,stride=1,padding=1,bias=False),
      torch.nn.BatchNorm2d(n//8),
      torch.nn.ReLU(inplace=True),
      torch.nn.Upsample(scale_factor=2),
      torch.nn.Conv2d(n//8,n//16,kernel_size=3,stride=1,padding=1,bias=False),
      torch.nn.ReLU(inplace=True)
    )
    self.to_224 = torch.nn.Sequential(
      torch.nn.Conv2d(n,n,kernel_size=5,stride=1,padding=1,bias=False),
      torch.nn.Upsample(scale_factor=2),
      torch.nn.Conv2d(n,n//2,kernel_size=3,stride=1,padding=1,bias=False),
      torch.nn.BatchNorm2d(n//2),
      torch.nn.ReLU(inplace=True),
      torch.nn.Upsample(scale_factor=2),
      torch.nn.Conv2d(n//2,n//4,kernel_size=3,stride=1,padding=1,bias=False),
      torch.nn.BatchNorm2d(n//4),
      torch.nn.ReLU(inplace=True),
      torch.nn.Upsample(scale_factor=2),
      torch.nn.Conv2d(n//4,n//8,kernel_size=3,stride=1,padding=1,bias=False),
      torch.nn.BatchNorm2d(n//8),
      torch.nn.ReLU(inplace=True),
      torch.nn.Upsample(scale_factor=2),
      torch.nn.Conv2d(n//8,n//16,kernel_size=3,stride=1,padding=1,bias=False),
      torch.nn.ReLU(inplace=True)
    )
    self.conv2seg = torch.nn.Conv2d(n//16,num_class,kernel_size=3,stride=1,padding=1,bias=True)

# ---------------------------------------------------------------
  def encoder(self, x): # returns magnitude spectrum, phase spectrum
    x = self.downsampler(x)
    sample_cnt = x.shape[-1]
    ideal_hop_length = sample_cnt / (self.spec_dim[0] - 1)

    if not ideal_hop_length.is_integer():
      print(f"WARNING, choose audio chunk size to be integer multiple of hop_length {self.hop_length}! != {ideal_hop_length}")
    
    X = torch.stft( input=x,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    window=torch.hann_window(self.win_length).to(x.device),
                    center=True,
                    hop_length=self.hop_length,
                    onesided=True,
                    return_complex=True)

    return torch.abs(X), torch.angle(X)

  # ---------------------------------------------------------------
  def decoder(self, X): #takes complex spectrum, returns audio
    x = torch.istft( input=X,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    window=torch.hann_window(self.win_length).to(X.device),
                    center=True,
                    hop_length=self.hop_length,
                    onesided=True,
                    return_complex=False)
    x = self.upsampler(x)
    return x

  # ---------------------------------------------------------------
  def forward(self, audio_in):
    # normalize the audio
    mean = torch.mean(audio_in)
    std = torch.std(audio_in)
    audio_in = (audio_in - mean) / (std + 1e-8)
    print(f"forward pass audio in shape: {audio_in.shape}")

    if torch.isnan(audio_in).any():
      print("input audio contains nan after normalization!!!")

    # get the magnitude and phase spectra from the encoder (STFT)
    mix_spec_in, phase_in = self.encoder(audio_in)

    print(f"self.encoder output shape: {mix_spec_in.shape}")

    if phase_in.shape[0] == 2:
      print(f"forward: adding phase 3rd channel")
      phase_in = torch.cat((phase_in, phase_in[0].unsqueeze(0)), dim=0)
    if mix_spec_in.shape[0] == 2:
      print(f"forward: adding mix sum mag as 3rd channel")
      mix_spec_sum = mix_spec_in.sum(dim=0,keepdim=True)
      mix_spec_in = torch.cat((mix_spec_in, mix_spec_sum), dim=0)

    #print(f"encoder output: mag {mix_spec.shape} phase {phase_in.shape}")
    mix_spec = 10*torch.log10(mix_spec_in + 1e-8)

    mix_spec = mix_spec.unsqueeze(0)
    print(f"mix_spec before dino pass: {mix_spec.shape}")
    with torch.no_grad():
        features = self.dinov2.forward_features(mix_spec)['x_norm_patchtokens']
    print(f"dino output features: {features.shape}")
    #x = self.selu(self.classlayer_224(features))
    #x = self.to_224(x)
    x = self.selu(self.classlayer_448(features))
    print(f"output of FeatureTransformer: 1 x.shape {x.shape} [num_channels, width, height]")
    x = self.to_448(x)
    print(f"head 2 x.shape after convolutional upsampler to 448 {x.shape} [num_channels, width, height]")
    pred_mask = self.conv2seg(x)
    print(f"pred mask {pred_mask.shape} output of conv2seg")
    pred_mask_upscaled = torch.nn.functional.interpolate(pred_mask, size=(448, 448), mode='bilinear')
    print(f"pred mask upscaled {pred_mask_upscaled.shape}")
    pred_filtered_mag = pred_mask_upscaled * mix_spec_in
    pred_spec = pred_filtered_mag * torch.cos(phase_in) + 1.0j * pred_filtered_mag * torch.sin(phase_in)
    print(f"pred spec {pred_spec.shape}")
    #print(f"pred spec shape : {pred_spec.shape}")
    pred_audio = self.decoder(pred_spec.squeeze(0))
    print(f"pred audio {pred_audio.shape}")
    #print(f"pred audio shape: {pred_audio.shape}")
    return pred_audio, mix_spec_in, phase_in, pred_filtered_mag, pred_mask_upscaled
