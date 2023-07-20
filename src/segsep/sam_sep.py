import torch
import torchaudio
import torchaudio.transforms as T
from transformers import SamModel, SamConfig
from utils import should_skip_chunk

# --------------------------------------------------------------------------------------------------
class SamWrapper(torch.nn.Module):
  def __init__(self,
               n_fft=2048,
               win_length=2047,
               spec_dim=(1024, 1024),
               sample_rate=44100,
               resample_rate=22050,
               saved_model_state_dict=None):
    super(SamWrapper, self).__init__()
    self.n_fft = n_fft
    self.win_length = win_length
    self.spec_dim = spec_dim
    self.sample_rate = sample_rate
    self.hop_length = self.n_fft // 4
    self.resample_rate = resample_rate
    self.downsampler = T.Resample(orig_freq=self.sample_rate, new_freq=self.resample_rate)
    self.upsampler = T.Resample(orig_freq=self.resample_rate, new_freq=self.sample_rate)

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

    sample_cnt = x.shape[-1]
    self.hop_length = int(sample_cnt / (2*self.spec_dim[0]))
    #print(f"calculated hop len: int({sample_cnt / (2*self.spec_dim[0])}) -> {self.hop_length}")
    x = self.downsampler(x)
    X = torch.stft( input=x,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    window=torch.hann_window(self.win_length).to(x.device),
                    center=True,
                    hop_length=self.hop_length,
                    onesided=True,
                    return_complex=True)
    #print(f"X.shape: {X.shape} hop_length: {self.hop_length}")
    #crop to shape
    #X = X[:, :self.spec_dim[0], :self.spec_dim[1]]

    #pad to shape
    padding_dims = (0, self.spec_dim[1] - X.shape[2], 0, self.spec_dim[0] - X.shape[1])
    X = F.pad(X, padding_dims, "constant", 0)
    #print(f"after padding: {X.shape}")
    return torch.abs(X), torch.angle(X)

  # ---------------------------------------------------------------
  def decoder(self, X): #takes complex spectrum, returns audio
    #print(f"decoder: input X shape : {X.shape}")
    X_padded = F.pad(X, (0,0,0,1), "constant", 0)
    x = torch.istft( input=X_padded,
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

    pred_masks = outputs.pred_masks.squeeze(1)
    upscaled_pred_mask = torch.nn.functional.interpolate(pred_masks, size=(1024,1024), mode='bicubic')

    # filter the predicted mask with the original mix spectrum magnitude
    pred_mag = torch.mul(upscaled_pred_mask, mix_spec_in)

    # resynthesize the estimated source audio
    pred = (pred_mag * torch.cos(phase_in)) + (1.0j * pred_mag * torch.sin(phase_in))
    #print(f"pred.shape: {pred.shape}")
    pred_audio = self.decoder(pred.squeeze())
    return pred_audio, mix_spec_in, phase_in, pred_mag, upscaled_pred_mask.squeeze()
  


def train(model, dataloader, optimizer, loss_fn, acc_fn, device):
  model.train()
  epoch_loss = 0
  epoch_acc = 0

  # automatic mixed precision scaler:
  scaler = torch.cuda.amp.GradScaler()

  for idx, (mix_audio, vocal_audio) in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()

    # Assumes that the dataloader returns tensors of size [num_samples]
    # and the audio length is larger than spec_len_in_s * sample_rate
    # Divide the audio into chunks of spec_len_in_s seconds
    spec_len_in_s = 5.0
    sample_rate = 44100

    # choose the hop_length so that the resulting spectrogram is precisely spec_dim[0] frames
    # -> make chunk_size a clean multiple of (2*spec_dim[0]) samples around spec_len_in_s seconds length
    chunk_cnt = int(spec_len_in_s * sample_rate / (2 * model.spec_dim[0]))
    chunk_size = 2 * model.spec_dim[0] * chunk_cnt
    real_spec_len_s = chunk_size / sample_rate
    #print(f"target chunk len in s: {spec_len_in_s} -> rounded to {chunk_cnt} frames of {2*model.spec_dim[0]} samples -> {real_spec_len_s}s")

    # add a 3rd channel to each audio tensor:
    vocal_audio_ch3=(vocal_audio[:,0]-vocal_audio[:,1]).unsqueeze(0)
    mix_audio_ch3=(mix_audio[:,0]-mix_audio[:,1]).unsqueeze(0)

    mix_audio = torch.cat((mix_audio, mix_audio_ch3), dim=1)
    vocal_audio = torch.cat((vocal_audio, vocal_audio_ch3), dim=1)

    track_loss = 0
    track_acc = 0
    chunk_cnt = 0
    skip_chunk_cnt = 0
    sample_cnt = mix_audio.shape[2] # [batch, channel, audio]

    mix_audio = mix_audio.squeeze().to(device)
    vocal_audio = vocal_audio.squeeze().to(device)
    for start_idx in range(0, sample_cnt, chunk_size):
      end_idx = start_idx + chunk_size
      mix_chunk = mix_audio[:, start_idx:end_idx]
      vocal_chunk = vocal_audio[:, start_idx:end_idx]
      #print(f"doing idx {start_idx}:{end_idx} - {float(end_idx)/mix_audio_3.shape[1]:.3f}")

      if should_skip_chunk(mix_chunk) or should_skip_chunk(vocal_chunk):
        skip_chunk_cnt += 1
        continue

      #print(f"mix chunk shape {mix_chunk.shape}")
      if torch.isnan(mix_chunk).any():
        print("input data contains nan!")
        mix_chunk = torch.nan_to_num(mix_chunk)

      # context for automatic mixed precision
      with torch.cuda.amp.autocast():
        pred_audio, mix_mag, mix_phase, pred_spec = model(mix_chunk)

        if torch.isnan(pred_audio).any():
          print("prediction contains nan!")
          pred_audio = torch.nan_to_num(pred_audio)
        trim_idx = min(pred_audio.shape[1], vocal_chunk.shape[1])

        # calculate loss/acc on SPEC
        vocal_spec, vocal_phase = model.encoder(vocal_chunk)
        loss = loss_fn(pred_spec.squeeze(), vocal_spec)
        acc = acc_fn(pred_spec, vocal_spec)

      epoch_loss += loss.item()
      track_loss += loss.item()

      epoch_acc += acc.item()
      track_acc += acc.item()
      chunk_cnt += 1

      # traditional backward pass
      #loss.backward()
      #optimizer.step()

      # backward pass with automatic mixed precision
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      # madre mia
      del pred_audio
      del loss
      del acc

    del mix_audio
    del vocal_audio
    torch.cuda.empty_cache()
    print(f"TRAIN track {idx}/{len(dataloader)} loss: {track_loss}, track acc: {track_acc:.4f}, skip chunk cnt: {skip_chunk_cnt}/{chunk_cnt}")
  return epoch_loss / len(dataloader), epoch_acc / len(dataloader)