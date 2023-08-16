import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
from transformers import SamModel, SamConfig
from segsep.utils import should_skip_chunk

# --------------------------------------------------------------------------------------------------
class SamWrapper(torch.nn.Module):
  def __init__(self,
               spec_dim=(1024, 1024),
               sample_rate=44100,
               resample_rate=22050,
               saved_model_state_dict=None):
    super().__init__()
    
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

    if saved_model_state_dict == None:
      self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
    else:
      config = SamConfig() # dummy config
      self.sam_model = SamModel(config)
      self.sam_model.load_state_dict(saved_model_state_dict)
      

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

    return X

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
  # returns predicted vocals:
  def forward(self, 
              audio_in, 
              input_points=None, 
              input_boxes=None, 
              upscale_mask=False, 
              debug=False):
    # normalize the audio
    mean = torch.mean(audio_in)
    std = torch.std(audio_in)
    audio_in = (audio_in - mean) / (std + 1e-8)

    if torch.isnan(audio_in).any():
      print("forward(): input audio contains nan after normalization!!!")
    
    # get the magnitude and phase spectra from the encoder (STFT)
    X = self.encoder(audio_in)
    input_spec_shape = X.shape

    mix_spec_in = torch.abs(X)
    phase_in = torch.angle(X)

    if debug:
      print(f"forward(): mix_spec {mix_spec_in.shape}, phase_in {phase_in.shape}")
    # make sure the tensors are in the correct shape for the backbone : 1024,1024
    if(mix_spec_in.shape[0] != 1024 or mix_spec_in.shape[1] != 1024):
      if debug: 
        print(f"forward(): interpolating input mix spec from {mix_spec_in.shape} to (3,1024,1024)")
      mix_spec_in = torch.nn.functional.interpolate(mix_spec_in.unsqueeze(0), size=(1024,1024), mode='bilinear').squeeze(0)
    outputs = self.sam_model(pixel_values=mix_spec_in.unsqueeze(0),
                             input_points=input_points,
                             input_boxes=input_boxes,
                             multimask_output=True)

    #print(f"sam_model iou_scores: {outputs.iou_scores.shape}; {outputs.iou_scores}")
    #img_embeddings = self.sam_model.vision_encoder(mix_spec_in.unsqueeze(0))
    #print(f"img_embeddings.last_hidden_state.shap: {img_embeddings.last_hidden_state.shape}")
    #mask = self.sam_model.mask_decoder(img_embeddings.last_hidden_state)
    #print(f"mask decoder output: {mask}")
    
    pred_masks = outputs.pred_masks.squeeze(1)
    if debug:
      print(f"forward(): sam_model outputs pred_masks shape: {outputs.pred_masks.shape}; pred_masks.shape: {pred_masks.shape}")
    
    
    if upscale_mask:
      upscale_size = (mix_spec_in.shape[1], mix_spec_in.shape[2])
      pred_masks = torch.nn.functional.interpolate(pred_masks, size=upscale_size, mode='bilinear')
      # filter the predicted mask with the original mix spectrum magnitude
      if debug:
        print(f"forward(): multiplying pred mask {pred_masks.shape} and mix spec {mix_spec_in.shape}")
      pred_mag = torch.mul(pred_masks, mix_spec_in)
      if debug:
        print(f"forward(): upscaling predicted mask from {pred_masks.shape} to {pred_masks.shape}")

    else: #downscale the mix_spec_in to pred_mag's size 
      kernel_size = (2,2) # ...first do a average pooling
      mix_spec_avg_pool = torch.nn.functional.avg_pool2d(mix_spec_in, kernel_size)
      downscale_size = (pred_masks.shape[2], pred_masks.shape[3])
      
      downscaled_mix_spec = torch.nn.functional.interpolate(mix_spec_in.unsqueeze(0), size=downscale_size, mode='bilinear').squeeze(0)

      if debug:
        print(f"forward(): downscaling mix spec from {mix_spec_in.shape} -> avg pool {mix_spec_avg_pool.shape} -> interpolate {downscaled_mix_spec.shape}")

      pred_mag = torch.mul(pred_masks, downscaled_mix_spec)
      
    # resynthesize the estimated source audio
    if pred_mag.squeeze().shape != input_spec_shape:
      if debug:
        print(f"forward(): resizing predicted mag spec from {pred_mag.shape} to {input_spec_shape}")
      pred_mag = torch.nn.functional.interpolate(pred_mag, size=(input_spec_shape[1], input_spec_shape[2]), mode='bilinear')

    pred = (pred_mag * torch.cos(phase_in)) + (1.0j * pred_mag * torch.sin(phase_in))
    
    if debug:
      print(f"forward(): predicted spectrum shape: {pred.shape}")
    pred_audio = self.decoder(pred.squeeze())
    return pred_audio, mix_spec_in, phase_in, pred_mag, pred_masks.squeeze(), outputs.iou_scores.squeeze()