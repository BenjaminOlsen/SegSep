import torch
from tqdm.auto import tqdm
from segsep.utils import should_skip_chunk
from torch.utils.data import SubsetRandomSampler

# --------------------------------------------------------------------------------------------------
def train(model, dataloader, optimizer, loss_fn, acc_fn, device, verbose=False):
  model.train()
  epoch_loss = 0
  epoch_acc = 0

  # automatic mixed precision scaler:
  scaler = torch.cuda.amp.GradScaler()

  # choose chunks of audio of chunk_size samples such that
  # each chunk results in a STFT of model.spec_dim[0] time bins
  hop_len = model.hop_length
  hop_cnt = model.spec_dim[0]
  chunk_size = model.input_chunk_size
  print(f"choosing audio of {chunk_size} samples -> {chunk_size/model.sample_rate:.5f}s")

  for idx, (mix_audio, source_audio) in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()
    # add a 3rd channel to each audio tensor:
    if source_audio.shape[1] == 2:
      source_audio_ch3=(source_audio[:,0]-source_audio[:,1]).unsqueeze(0)
      source_audio = torch.cat((source_audio, source_audio_ch3), dim=1)
    elif source_audio.shape[1] == 1:
        source_audio = torch.cat((source_audio, source_audio, source_audio), dim=1)
    if mix_audio.shape[1] == 2:
      mix_audio_ch3=(mix_audio[:,0]-mix_audio[:,1]).unsqueeze(0)
      mix_audio = torch.cat((mix_audio, mix_audio_ch3), dim=1)
    elif mix_audio.shape[1] == 1:
      mix_audio = torch.cat((mix_audio, mix_audio, mix_audio), dim=1)


    track_loss = 0
    track_acc = 0
    chunk_cnt = 0
    skip_chunk_cnt = 0
    sample_cnt = mix_audio.shape[2] # [batch, channel, audio]

    mix_audio = mix_audio.squeeze().to(device)
    source_audio = source_audio.squeeze().to(device)
    # omit last chunk if its size would be less than chunk_size
    for start_idx in range(0, sample_cnt-chunk_size+1, chunk_size):
      end_idx = start_idx + chunk_size
      mix_chunk = mix_audio[:, start_idx:end_idx]
      source_chunk = source_audio[:, start_idx:end_idx]
      #print(f"doing idx {start_idx}:{end_idx} - {float(end_idx)/mix_audio_3.shape[1]:.3f}")

      if should_skip_chunk(mix_chunk) or should_skip_chunk(source_chunk):
        skip_chunk_cnt += 1
        continue

      #print(f"mix chunk shape {mix_chunk.shape}")
      if torch.isnan(mix_chunk).any():
        print("input data contains nan!")
        mix_chunk = torch.nan_to_num(mix_chunk)

      # context for automatic mixed precision
      with torch.cuda.amp.autocast():
        pred_audio, mix_spec_in, phase_in, pred_mag, upscaled_pred_mask, iou_scores = model(mix_chunk)

        if torch.isnan(pred_audio).any():
          print("prediction contains nan!")
          continue
          #pred_audio = torch.nan_to_num(pred_audio)

        trim_idx = min(pred_audio.shape[1], source_chunk.shape[1])

        # calculate loss/acc on SPEC
        source_spec, source_phase = model.encoder(source_chunk)
        loss = loss_fn(pred_mag.squeeze(), source_spec)
        acc = acc_fn(pred_mag, source_spec)

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
    del source_audio

    torch.cuda.empty_cache()
    if verbose:
      print(f"TRAIN track {idx}/{len(dataloader)} loss: {track_loss}, track acc: {track_acc:.4f}, skip chunk cnt: {skip_chunk_cnt}/{chunk_cnt}")
  return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

# --------------------------------------------------------------------------------------------------
def validate(model, dataloader, loss_fn, acc_fn, device, subset_proportion=0.1, verbose=False):
  model.eval()
  epoch_loss = 0
  epoch_acc = 0

  with torch.inference_mode():

    subset_sample_cnt = int(len(dataloader) * subset_proportion)
    subset_indices = torch.randperm(len(dataloader))[:subset_sample_cnt]
    subset_sampler = SubsetRandomSampler(subset_indices)
    subset_dataloader = torch.utils.data.DataLoader(dataloader.dataset,
                                                    batch_size=dataloader.batch_size,
                                                    sampler=subset_sampler,
                                                    num_workers=dataloader.num_workers,
                                                    pin_memory=dataloader.pin_memory)

    # choose chunks of audio of chunk_size samples such that
    # each chunk results in a STFT of model.spec_dim[0] time bins
    hop_len = model.hop_length
    hop_cnt = model.spec_dim[0]
    chunk_size = model.input_chunk_size

    #print(f"choosing audio of {chunk_size} samples -> {chunk_size/model.resample_rate:.5f}s")

    for idx, (mix_audio, source_audio) in enumerate(tqdm(subset_dataloader)):
      # add a 3rd channel to each audio tensor:
      if source_audio.shape[1] == 2:
        source_audio_ch3=(source_audio[:,0]-source_audio[:,1]).unsqueeze(0)
        source_audio = torch.cat((source_audio, source_audio_ch3), dim=1)
      elif source_audio.shape[1] == 1:
          source_audio = torch.cat((source_audio, source_audio, source_audio), dim=1)
      if mix_audio.shape[1] == 2:
        mix_audio_ch3=(mix_audio[:,0]-mix_audio[:,1]).unsqueeze(0)
        mix_audio = torch.cat((mix_audio, mix_audio_ch3), dim=1)
      elif mix_audio.shape[1] == 1:
        mix_audio = torch.cat((mix_audio, mix_audio, mix_audio), dim=1)

      track_loss = 0
      track_acc = 0
      chunk_cnt = 0
      skip_chunk_cnt = 0
      sample_cnt = mix_audio.shape[2] # [batch, channel, audio]

      mix_audio = mix_audio.squeeze().to(device)
      source_audio = source_audio.squeeze().to(device)
      # omit last chunk if its size would be less than chunk_size
      for start_idx in range(0, sample_cnt-chunk_size+1, chunk_size):
        end_idx = start_idx + chunk_size
        mix_chunk = mix_audio[:, start_idx:end_idx]
        source_chunk = source_audio[:, start_idx:end_idx]
        #print(f"doing idx {start_idx}:{end_idx} - {float(end_idx)/mix_audio.shape[1]:.3f}")

        if should_skip_chunk(mix_chunk) or should_skip_chunk(source_chunk):
          skip_chunk_cnt += 1
          continue

        #print(f"mix chunk shape {mix_chunk.shape}")
        if torch.isnan(mix_chunk).any():
          print("input data contains nan!")
          continue
          #mix_chunk = torch.nan_to_num(mix_chunk)
        pred_audio, mix_spec_in, phase_in, pred_mag, upscaled_pred_mask, iou_scores = model(mix_chunk)
        if torch.isnan(pred_audio).any():
          print("predicted audio contains nan!")
          continue
        trim_idx = min(pred_audio.shape[1], source_chunk.shape[1])

        # calculate loss/acc on SPEC
        source_spec, source_phase = model.encoder(source_chunk)
        loss = loss_fn(pred_mag.squeeze(), source_spec)
        acc = acc_fn(pred_mag, source_spec)

        epoch_loss += loss.item()
        track_loss += loss.item()

        epoch_acc += acc.item()
        track_acc += acc.item()

        chunk_cnt += 1

        # madre mia
        del pred_audio
        del loss
      del mix_audio
      del source_audio
      torch.cuda.empty_cache()
      if verbose:
        print(f"TEST track {idx}/{len(subset_dataloader)} loss: {track_loss:.8f}, track acc: {track_acc:.4f}, skip chunk cnt: {skip_chunk_cnt}/{chunk_cnt}")
  return epoch_loss / len(subset_dataloader), epoch_acc / len(subset_dataloader)