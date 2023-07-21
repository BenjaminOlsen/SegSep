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
      print(f"WARNING, noninteger input chunk size, choose your samplerate, resample rate to divide cleanly!")
    print(f"SamWrapper model n_fft: {self.n_fft}, win len: {self.win_length}, hop len: {self.hop_length}, sample/resample: {self.sample_rate / self.resample_rate} -> input_chunk_size {self.input_chunk_size}")
    
    self.input_chunk_size = int(self.input_chunk_size)

    if saved_model_state_dict == None:
      self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
    else:
      config = SamConfig() # dummy config
      self.sam_model = SamModel()

  # ---------------------------------------------------------------
  def encoder(self, x): # returns magnitude spectrum, phase spectrum

    # try to choose the hop_length so that the spectrogram results close
    # to self.spec_dim[0] frames
    #print(f"encoder arg: {x.shape}")
    
    x = self.downsampler(x)
    #print(f"encoder downsample output shape {x.shape}")
    sample_cnt = x.shape[-1]
    #print(f"sample cnt {sample_cnt} / hop_len {self.hop_length} = {sample_cnt/self.hop_length}")

    ideal_hop_length = sample_cnt / (self.spec_dim[0] - 1)

    if not ideal_hop_length.is_integer():
      print(f"WARNING, choose audio chunk size to be integer multiple of hop_length {self.hop_length}! != {ideal_hop_length}")

    #print(f"calculated hop len: int({self.hop_length}) -> {int(self.hop_length)}")
    #self.hop_length = int(self.hop_length)
    #ideal_sample_cnt = self.hop_length * (self.spec_dim[0]-1) + self.win_length
    #print(f"ideal_sample_cnt: {ideal_sample_cnt}")
    
    X = torch.stft( input=x,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    window=torch.hann_window(self.win_length).to(x.device),
                    center=True,
                    hop_length=self.hop_length,
                    onesided=True,
                    return_complex=True)
    #print(f"after stft X.shape: {X.shape} hop_length: {self.hop_length}")
    
    #crop to shape
    #X = X[:, :self.spec_dim[0], :self.spec_dim[1]]
    #pad to shape
    #padding_dims = (0, self.spec_dim[1] - X.shape[2], 0, self.spec_dim[0] - X.shape[1])
    #X = F.pad(X, padding_dims, "constant", 0)
    #print(f"after padding: {X.shape}")
    return torch.abs(X), torch.angle(X)

  # ---------------------------------------------------------------
  def decoder(self, X): #takes complex spectrum, returns audio
    #print(f"decoder: input X shape : {X.shape}")
    #X_padded = F.pad(X, (0,0,0,1), "constant", 0)
    x = torch.istft( input=X,#X_padded,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    window=torch.hann_window(self.win_length).to(X.device),
                    center=True,
                    hop_length=self.hop_length,
                    onesided=True,
                    return_complex=False)
    #print(f"decoder: upsampler input shape {x.shape}")
    x = self.upsampler(x)
    #print(f"decoder upsampler output shape {x.shape}")
    return x

  # ---------------------------------------------------------------
  # returns predicted vocals:
  def forward(self, audio_in):
    # normalize the audio
    mean = torch.mean(audio_in)
    std = torch.std(audio_in)
    audio_in = (audio_in - mean) / (std + 1e-8)

    if torch.isnan(audio_in).any():
      print("input audio contains nan after normalization!!!")
    # get the magnitude and phase spectra from the encoder (STFT)
    mix_spec_in, phase_in = self.encoder(audio_in)

    #print(f"mix_spec {mix_spec.shape}, phase_in {phase_in.shape}")
    # make sure the tensors are in the correct shape for the backbone
    outputs = self.sam_model(pixel_values=mix_spec_in.unsqueeze(0),
                             input_boxes=None,
                             multimask_output=False)

    #img_embeddings = self.sam_model.vision_encoder(mix_spec_in.unsqueeze(0))
    #print(f"img_embeddings.last_hidden_state.shap: {img_embeddings.last_hidden_state.shape}")
    #mask = self.sam_model.mask_decoder(img_embeddings.last_hidden_state)
    #print(f"mask decoder output: {mask}")
    
    #print(f"sam_model outputs pred_masks shape: {outputs.pred_masks.shape}")
    pred_masks = outputs.pred_masks.squeeze(1)
    upscaled_pred_mask = torch.nn.functional.interpolate(pred_masks, size=(1024,1024), mode='bicubic')

    # filter the predicted mask with the original mix spectrum magnitude
    pred_mag = torch.mul(upscaled_pred_mask, mix_spec_in)

    # resynthesize the estimated source audio
    pred = (pred_mag * torch.cos(phase_in)) + (1.0j * pred_mag * torch.sin(phase_in))
    #print(f"pred.shape: {pred.shape}")
    pred_audio = self.decoder(pred.squeeze())
    return pred_audio, mix_spec_in, phase_in, pred_mag, upscaled_pred_mask.squeeze()