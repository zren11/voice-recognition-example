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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


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
    random.seed(41)
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


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        loss_fn,
        scheduler,
        device,
        total_epochs,
        exp_path=None,
        rank=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.total_epochs = total_epochs
        self.best_val_accuracy = 0.0
        self.step = 0
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.total_steps = len(self.train_loader) * self.total_epochs
        if exp_path is None:
            self.exp_path = f"runs/exp-{self.timestamp}"
        else:
            self.exp_path = exp_path
        print(f"Experiment path: {self.exp_path}")

        self.rank = rank
        self.is_main_process = self.rank == 0

        if self.is_main_process:
            self.writer = SummaryWriter(f"{self.exp_path}/logs")
        else:
            self.writer = None

    def _train_one_epoch(self, epoch):
        self.train_loader.sampler.set_epoch(epoch)
        self.model.train()  # 确保模型在训练模式
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (spectrograms, labels) in enumerate(self.train_loader):
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)

            # 1. PREDICT: Pass data through the model
            predictions = self.model(spectrograms)

            # 2. COMPARE: Calculate the error
            loss = self.loss_fn(predictions, labels)

            # 3. ADJUST: Update the model's weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update tracking variables
            epoch_loss += loss.item()
            num_batches += 1
            self.step += 1

            # Log to TensorBoard every step
            if self.is_main_process:
                self.writer.add_scalar("Loss/Train", loss.item(), self.step)
                # Print progress every 10 steps or at the end of each epoch
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(
                    self.train_loader
                ):
                    avg_loss = epoch_loss / num_batches
                    print(
                        f"Epoch {epoch+1:2d}/{self.total_epochs} | Step {self.step:4d}/{self.total_steps} | "
                        f"Batch {batch_idx+1:3d}/{len(self.train_loader)} | "
                        f"Loss: {loss.item():.6f} | Avg Loss: {avg_loss:.6f}"
                    )

        # Log epoch average loss to TensorBoard
        avg_epoch_loss = epoch_loss / num_batches
        if self.is_main_process:
            self.writer.add_scalar("Loss/Epoch_Avg", avg_epoch_loss, epoch + 1)
            # Print epoch summary
            print(
                f"Epoch {epoch+1:2d}/{self.total_epochs} completed | Avg Loss: {avg_epoch_loss:.6f}"
            )
            print("-" * 80)

    def _validate_one_epoch(self, epoch, loader=None):
        if hasattr(loader, "sampler") and isinstance(
            loader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            loader.sampler.set_epoch(epoch)

        self.model.eval()  # 1. 切换到评估模式
        # These are calculated locally on each process
        local_val_loss = 0
        local_correct = 0
        local_total = 0

        if self.is_main_process:
            print("  Starting validation...")

        with torch.no_grad():  # 2. 在此代码块内不计算梯度
            for batch_idx, (spectrograms, labels) in enumerate(loader):
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)

                # 只进行预测和计算损失
                predictions = self.model(spectrograms)
                loss = self.loss_fn(predictions, labels)

                local_val_loss += loss.item()

                # 计算准确率
                _, predicted_labels = torch.max(predictions.data, 1)
                local_total += labels.size(0)
                local_correct += (predicted_labels == labels).sum().item()

                # 每10个batch打印一次验证进度
                if self.is_main_process and (
                    (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader)
                ):
                    current_accuracy = 100 * local_correct / local_total
                    print(
                        f"    Validation Batch {batch_idx+1:3d}/{len(loader)} | "
                        f"Current Accuracy: {current_accuracy:.2f}%"
                    )
        # --- DDP Synchronization Step ---
        if dist.is_initialized():
            # If we are in DDP, we need to aggregate results from all processes
            metrics = torch.tensor([local_correct, local_total, local_val_loss]).to(
                self.device
            )
            # Sum the values from all GPUs
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            # Get the total values
            total_correct = metrics[0].item()
            total_samples = metrics[1].item()
            total_loss = metrics[2].item()
        else:
            # If not in DDP, the local values are the total values
            total_correct = local_correct
            total_samples = local_total
            total_loss = local_val_loss

        if self.is_main_process:
            if dist.is_initialized():
                # In DDP, the total number of batches is len(loader) * world_size
                world_size = dist.get_world_size()
                avg_loss = total_loss / (len(loader) * world_size)
            else:
                avg_loss = total_loss / len(loader)

            val_accuracy = 100 * total_correct / total_samples

            print(
                f"  Validation completed | Loss: {avg_loss:.6f} | Accuracy: {val_accuracy:.2f}%"
            )

            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                print(
                    f"New best validation accuracy: {self.best_val_accuracy:.2f}%. Saving model..."
                )
                model_folder = f"{self.exp_path}/models"
                os.makedirs(model_folder, exist_ok=True)
                torch.save(
                    self.model.state_dict(), f"{model_folder}/best-epoch{epoch}.pth"
                )

                self.writer.add_scalar("Loss/Val", total_loss, epoch + 1)
                self.writer.add_scalar("Accuracy/Val", val_accuracy, epoch + 1)

    def train(self):
        print("Starting training...")
        for epoch in range(self.total_epochs):
            self._train_one_epoch(epoch)
            self._validate_one_epoch(epoch, self.val_loader)
            if self.scheduler:
                self.scheduler.step()
            if self.is_main_process:
                print("-" * 80)

        if self.is_main_process:
            self.writer.close()
            print("Training complete!")

    def test(self):
        if self.is_main_process:
            print("Starting test...")
        self.model.eval()  # 1. 切换到评估模式
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # 2. 在此代码块内不计算梯度
            for batch_idx, (spectrograms, labels) in enumerate(self.test_loader):
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)

                # 只进行预测和计算损失
                predictions = self.model(spectrograms)
                loss = self.loss_fn(predictions, labels)

                val_loss += loss.item()

                # 计算准确率
                _, predicted_labels = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()

                if self.is_main_process:
                    # 每10个batch打印一次验证进度
                    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(
                        self.test_loader
                    ):
                        current_accuracy = 100 * correct / total
                        print(
                            f"    Test Batch {batch_idx+1:3d}/{len(self.test_loader)} | "
                            f"Current Accuracy: {current_accuracy:.2f}%"
                        )

        avg_loss = val_loss / len(self.test_loader)
        val_accuracy = 100 * correct / total

        if self.is_main_process:
            print(
                f"  Test completed | Loss: {avg_loss:.6f} | Accuracy: {val_accuracy:.2f}%"
            )

            print("Test complete!")


def setup_ddp():
    """初始化DDP进程组"""
    # 这行是关键：为当前进程绑定唯一的GPU
    if (
        "LOCAL_RANK" not in os.environ
        or "RANK" not in os.environ
        or "WORLD_SIZE" not in os.environ
    ):
        print("LOCAL_RANK, RANK, and WORLD_SIZE is not set, will skip using DDP")
        return torch.device("cuda") if torch.cuda.is_available() else "cpu", 0, 0
    print(
        f"LOCAL_RANK: {os.environ['LOCAL_RANK']}, RANK: {os.environ['RANK']}, WORLD_SIZE: {os.environ['WORLD_SIZE']}"
    )

    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)

    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")

    rank = torch.distributed.get_rank()
    print(
        f"Rank(dist.get_rank()): {rank}, Rank(os.environ['RANK']): {os.environ['RANK']}, Local Rank(os.environ['LOCAL_RANK']): {local_rank}"
    )

    return device, local_rank, rank


def cleanup_ddp():
    """销毁进程组"""
    dist.destroy_process_group()


def train():
    # 这行代码会自动选择GPU（如果可用），否则退回到CPU
    device, local_rank, rank = setup_ddp()

    print(f"Using device: {device}")

    # Instantiate the Dataset and DataLoader
    print("start loading dataset")
    audio_info, label_map, label_map_reverse = load_data(
        "/data/SpeechCommands/speech_commands_v0.02"
    )
    debug = True
    # comment out this line if you want to train on a smaller dataset for a faster debugging purpose
    if debug:
        audio_info = audio_info[:10000]

    audio_info_training, audio_info_validation, audio_info_test = split_data(audio_info)
    train_dataset = SpeechCommandsDataset(
        audio_info_training, label_map, label_map_reverse
    )

    val_dataset = SpeechCommandsDataset(
        audio_info_validation, label_map, label_map_reverse
    )
    test_dataset = SpeechCommandsDataset(audio_info_test, label_map, label_map_reverse)
    print("dataset loaded")

    print("start init data loader")

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_fn_spectrogram,
    )
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn_spectrogram,
    )
    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collate_fn_spectrogram,
    )

    print("data loader initialized")

    # Instantiate the Model, Loss Function, and Optimizer
    model = AudioTransformer().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Experiment timestamp: {timestamp}")

    # --- 2. 创建并启动 Trainer ---
    total_epochs = 2 if debug else 30
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        loss_fn,
        scheduler,
        device,
        total_epochs,
        rank=rank,
    )
    trainer.train()

    # --- 3. (可选) 最终测试 ---
    trainer.test()  # 你可以为 Trainer 添加一个 .test() 方法

    cleanup_ddp()

    if rank == 0:
        print("Training complete!")
        print("To view TensorBoard, run: tensorboard --logdir=runs")


if __name__ == "__main__":
    train()
