import torch
import argparse

from statistics import mean
from datetime import datetime
from datasets import AudioPairDataset
from segsep.sam_sep import SamWrapper
from segsep.train_validate import train, validate
from segsep.loss_acc import si_snr, LOGL2loss_freq


def main(args):
  test_audio_metadata_json = args.test_metadata
  train_audio_metadata_json = args.train_metadata
  fsd_train_location = args.train_dataset
  fsd_test_location = args.test_dataset

  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  model = SamWrapper(spec_dim=(1024,1024),
                    sample_rate=44100,
                    resample_rate=22050,
                    saved_model_state_dict=None).to(device)

  learning_rate = 1e-5
  optimizer = torch.optim.Adam(model.sam_model.vision_encoder.parameters(), lr=learning_rate, weight_decay=0)
  
  centroid_diff_hz = 2000.0

  train_dataset = AudioPairDataset(audio_dir=fsd_train_location,
                                   json_path=train_audio_metadata_json,
                                   centroid_diff_hz=centroid_diff_hz,
                                   min_duration_s = model.input_chunk_size/model.resample_rate,
                                   dummy_mode=False)
  
  test_dataset = AudioPairDataset(audio_dir=fsd_test_location,
                                   json_path=test_audio_metadata_json,
                                   centroid_diff_hz=centroid_diff_hz,
                                   min_duration_s = model.input_chunk_size/model.resample_rate,
                                   dummy_mode=False)
  
  train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size=1,
                                                  num_workers=0,
                                                  shuffle=True)
  
  test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  num_workers=0,
                                                  shuffle=True)
  # Initialize stats tracking
  train_losses = []
  test_losses = []
  train_accs = []
  test_accs = []

  num_epochs = 35
  # Training and validation loop
  for epoch in range(num_epochs):
    print(f"starting epoch {epoch}")
    # Training

    train_loss, train_acc = train(model=model,
                                  dataloader=train_dataloader,
                                  optimizer=optimizer,
                                  loss_fn=LOGL2loss_freq,
                                  acc_fn=si_snr,
                                  device=device)

    mean_train_loss = 0 if len(train_losses) == 0 else mean(train_losses)
    mean_train_acc = 0 if len(train_accs) == 0 else mean(train_accs)

    delta_loss = train_loss - mean_train_loss
    delta_acc = train_acc - mean_train_acc
    print(f"epoch {epoch} train done -> avg train_loss: {train_loss} (delta {delta_loss}), acc: {train_acc} (delta {delta_acc})")
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    print(f"mean train_loss so far: {mean(train_losses)}, acc: {mean(train_accs)}")

    # Validation
    test_loss, test_acc = validate(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=LOGL2loss_freq,
                                    acc_fn=si_snr,
                                    device=device)


    mean_test_loss = 0 if len(test_losses) == 0 else mean(test_losses)
    mean_test_acc = 0 if len(test_accs) == 0 else mean(test_accs)
    delta_loss = test_loss - mean_test_loss
    delta_acc = test_acc - mean_test_acc
    print(f"epoch {epoch} test done -> avg test_loss: {test_loss} (delta {delta_loss}), acc: {test_acc} (delta {delta_acc})")
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(f"mean test_loss so far: {mean(test_losses)}, acc: {mean(test_accs)}")

    # Print stats
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {train_loss}, Train Acc: {train_acc}")
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}")

    # Save checkpoint every 3 epochs
    if (epoch + 1) % 3 == 0:
      now = datetime.now()
      date_time = now.strftime("%m-%d-%y_%H:%M:%S")
      checkpoint_path = f'/content/drive/MyDrive/models/{model_name}-{loss_name}-lr{learning_rate}-{epoch}-epoch_{date_time}.pth'
      print(f"SAVING checkpoint: {checkpoint_path}")
      torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'train_loss': train_loss,
          'test_loss': test_loss,
          'train_acc': train_acc,
            'test_acc': test_acc
            }, checkpoint_path)
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="train the SAM image encoder")
  parser.add_argument("--test_metadata", type=str, default="eval_audio_metadata.json",
                      help='location of the TEST audio metadata generated by segsep.datasets.generate_audio_metadata')
  parser.add_argument("--train_metadata", type=str, default="dev_audio_metadata.json",
                      help='location of the TRAIN audio metadata generated by segsep.datasets.generate_audio_metadata')
  parser.add_argument("--train_dataset", type=str, default="FSD50K.dev_audio",
                      help='location of the Free Sound Data dataset folder for trainig')
  parser.add_argument("--test_dataset", type=str, default="FSD50k.eval_audio",
                      help='location of the Free Sound Data dataset folder for testing')
  args = parser.parse_args()

  main(args)
