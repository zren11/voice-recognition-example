# 1. IMPORTS
# ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os  # To navigate file paths
import json
from torch.utils.tensorboard import SummaryWriter
import random
from datetime import datetime

# 这行代码会自动选择GPU（如果可用），否则退回到CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 2. THE DATASET CLASS
# ---
# Its job: Load an audio file, convert it to a spectrogram, and return it with its numerical label.
def load_data(data_path):
    audio_info = []
    label_map = {}
    label_map_reverse = {}
    # Walk through the data directory to find all audio files
    full_data_path = os.path.join(os.path.dirname(__file__), data_path)

    # Get all subdirectories (word labels), excluding special directories
    labels = []
    for item in os.listdir(full_data_path):
        item_path = os.path.join(full_data_path, item)
        if os.path.isdir(item_path) and not item.startswith("_") and item != "LICENSE":
            labels.append(item)

    # Sort labels for consistent mapping
    labels.sort()

    # Create label to integer mapping
    label_map = {label: idx for idx, label in enumerate(labels)}
    label_map_reverse = {idx: label for label, idx in enumerate(labels)}

    # Collect all audio files
    for label in labels:
        label_dir = os.path.join(full_data_path, label)
        for filename in os.listdir(label_dir):
            if filename.endswith(".wav"):
                # Store relative path from voice-recognition folder
                relative_path = os.path.join(data_path, label, filename)
                audio_info.append({"filename": relative_path, "label": label})
    return audio_info, label_map, label_map_reverse


def split_data(audio_info):
    random.seed(42)
    random.shuffle(audio_info)
    print(f"audio info length: {len(audio_info)}")
    train_size = int(len(audio_info) * 0.7)
    val_size = int(len(audio_info) * 0.2)
    test_size = len(audio_info) - train_size - val_size
    print(f"train size: {train_size}, val size: {val_size}, test size: {test_size}")

    audio_info_training = audio_info[:train_size]
    audio_info_validation = audio_info[train_size : train_size + val_size]
    audio_info_test = audio_info[train_size + val_size :]
    return audio_info_training, audio_info_validation, audio_info_test


class SpeechCommandsDataset(Dataset):
    def __init__(self, audio_info, label_map, label_map_reverse):
        self.audio_info = audio_info
        self.label_map = label_map
        self.label_map_reverse = label_map_reverse
        self.transform = torchaudio.transforms.MelSpectrogram(n_mels=128)

        print(json.dumps(self.audio_info[0:3], indent=4))
        print(self.label_map)

    def __len__(self):
        return len(self.audio_info)

    def __getitem__(self, index):
        audio_info = self.audio_info[index]

        # Build full path to audio file
        audio_path = os.path.join(os.path.dirname(__file__), audio_info["filename"])

        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Transform to spectrogram
        spectrogram = self.transform(waveform)
        # print(f"Spectrogram shape: {spectrogram.shape}")

        # Get the numerical label from the word label
        label = self.label_map[audio_info["label"]]

        return spectrogram, label


def collate_fn_spectrogram(batch):
    # batch is a list of tuples (spectrogram, label)

    # Let's set our target length
    target_length = 81

    spectrograms = []
    labels = []

    # Loop through each item in the batch
    for spec, label in batch:
        # spec shape is (1, num_features, time)
        current_length = spec.shape[2]

        # --- Padding or Truncating ---
        if current_length < target_length:
            # Pad with zeros if it's too short
            padding_needed = target_length - current_length
            # torch.pad takes (data, (pad_left, pad_right, pad_top, pad_bottom, ...))
            spec = torch.nn.functional.pad(spec, (0, padding_needed))
        elif current_length > target_length:
            # Truncate if it's too long
            spec = spec[:, :, :target_length]

        spectrograms.append(spec)
        labels.append(label)

    # Stack them into a single batch tensor
    spectrograms_batch = torch.cat(spectrograms, dim=0)
    labels_batch = torch.tensor(labels)

    return spectrograms_batch, labels_batch


# 3. THE MODEL CLASS
# ---
# Its job: Define the Transformer architecture.
class AudioTransformer(nn.Module):
    def __init__(self, num_input_features=128, num_classes=35, dropout=0.1):
        super().__init__()
        # Using PyTorch's pre-built Transformer components
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_input_features, nhead=4, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(num_input_features, num_classes)

    def forward(self, spectrogram_batch):
        # Input shape needs to be (batch, time, features) for batch_first=True
        # Spectrograms are often (batch, features, time), so we might need to permute
        x = spectrogram_batch.permute(0, 2, 1)

        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Average over the time dimension
        predictions = self.output_layer(x)
        return predictions


def validate_model(data_loader, model, loss_fn, device):
    model.eval()  # 1. 切换到评估模式
    val_loss = 0
    correct = 0
    total = 0

    print("  Starting validation...")

    with torch.no_grad():  # 2. 在此代码块内不计算梯度
        for batch_idx, (spectrograms, labels) in enumerate(data_loader):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # 只进行预测和计算损失
            predictions = model(spectrograms)
            loss = loss_fn(predictions, labels)

            val_loss += loss.item()

            # 计算准确率
            _, predicted_labels = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

            # 每10个batch打印一次验证进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
                current_accuracy = 100 * correct / total
                print(
                    f"    Validation Batch {batch_idx+1:3d}/{len(data_loader)} | "
                    f"Current Accuracy: {current_accuracy:.2f}%"
                )

    avg_loss = val_loss / len(data_loader)
    accuracy = 100 * correct / total

    print(f"  Validation completed | Loss: {avg_loss:.6f} | Accuracy: {accuracy:.2f}%")

    model.train()  # 验证结束后，别忘了切换回训练模式！
    return avg_loss, accuracy


# 4. THE TRAINING SCRIPT
# ---
# This block runs when you execute the python file.
if __name__ == "__main__":
    # Instantiate the Dataset and DataLoader
    print("start loading dataset")
    audio_info, label_map, label_map_reverse = load_data(
        "SpeechCommands/speech_commands_v0.02"
    )
    audio_info_training, audio_info_validation, audio_info_test = split_data(audio_info)
    dataset_training = SpeechCommandsDataset(
        audio_info_training, label_map, label_map_reverse
    )
    dataset_validation = SpeechCommandsDataset(
        audio_info_validation, label_map, label_map_reverse
    )
    dataset_test = SpeechCommandsDataset(audio_info_test, label_map, label_map_reverse)
    print("dataset loaded")

    print("start init data loader")
    data_loader_training = DataLoader(
        dataset_training,
        batch_size=512,
        shuffle=True,
        collate_fn=collate_fn_spectrogram,
    )
    data_loader_validation = DataLoader(
        dataset_validation,
        batch_size=512,
        shuffle=True,
        collate_fn=collate_fn_spectrogram,
    )
    data_loader_test = DataLoader(
        dataset_test, batch_size=512, shuffle=True, collate_fn=collate_fn_spectrogram
    )

    print("data loader initialized")

    # Instantiate the Model, Loss Function, and Optimizer
    model = AudioTransformer().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Experiment timestamp: {timestamp}")

    # Initialize TensorBoard writer with timestamp
    writer = SummaryWriter(f"runs/vr-exp-{timestamp}")

    # The Training Loop
    print("Starting training...")
    total_epochs = 20
    total_steps = len(data_loader_training) * total_epochs  # Total steps for all epochs
    step = 0

    best_val_accuracy = 0.0
    for epoch in range(total_epochs):  # An "epoch" is one full pass over the dataset
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (spectrograms, labels) in enumerate(data_loader_training):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # 1. PREDICT: Pass data through the model
            predictions = model(spectrograms)

            # 2. COMPARE: Calculate the error
            loss = loss_fn(predictions, labels)

            # 3. ADJUST: Update the model's weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update tracking variables
            epoch_loss += loss.item()
            num_batches += 1
            step += 1

            # Log to TensorBoard every step
            writer.add_scalar("Loss/Train", loss.item(), step)

            # Print progress every 10 steps or at the end of each epoch
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(
                data_loader_training
            ):
                avg_loss = epoch_loss / num_batches
                print(
                    f"Epoch {epoch+1:2d}/{total_epochs} | Step {step:4d}/{total_steps} | "
                    f"Batch {batch_idx+1:3d}/{len(data_loader_training)} | "
                    f"Loss: {loss.item():.6f} | Avg Loss: {avg_loss:.6f}"
                )

        # Log epoch average loss to TensorBoard
        avg_epoch_loss = epoch_loss / num_batches
        writer.add_scalar("Loss/Epoch_Avg", avg_epoch_loss, epoch + 1)
        # Print epoch summary
        print(
            f"Epoch {epoch+1:2d}/{total_epochs} completed | Avg Loss: {avg_epoch_loss:.6f}"
        )
        print("-" * 80)

        # Validate the model
        val_loss, val_accuracy = validate_model(
            data_loader_validation, model, loss_fn, device
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(
                f"New best validation accuracy: {best_val_accuracy:.2f}%. Saving model..."
            )
            torch.save(model.state_dict(), f"best_model_{timestamp}_epoch{epoch}.pth")
        writer.add_scalar("Loss/Val", val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch + 1)
        scheduler.step()

    print("Running final test...")
    test_loss, test_accuracy = validate_model(data_loader_test, model, loss_fn, device)
    print(
        f"Final Test Results | Test Loss: {test_loss:.6f} | Test Accuracy: {test_accuracy:.2f}%"
    )

    # Close TensorBoard writer
    writer.close()
    print("Training complete!")
    print("To view TensorBoard, run: tensorboard --logdir=runs")
