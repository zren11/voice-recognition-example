# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based speech recognition system that trains a Transformer model on the Speech Commands dataset v0.02. The project implements audio classification for single-word spoken commands using mel spectrograms as input features.

## Development Commands

**Training (Local):**
```bash
python train_model.py
```

**Training (Distributed):**
```bash
torchrun --nproc_per_node=2 train_model.py  # Multi-GPU local
# See example.ipynb for Kubeflow distributed training setup
```

**Prediction/Inference:**
```bash
python predict.py
```

**Data Preparation:**
```bash
python prepare-data.py
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**TensorBoard Visualization:**
```bash
tensorboard --logdir=runs
```

**Container Development:**
```bash
docker build -t speech-recognition .
kind create cluster --config kind-config.yaml  # Local K8s testing
```

## Architecture

### Core Components

**train_model.py** - Main training script containing:
- `SpeechCommandsDataset`: Custom PyTorch Dataset for loading and preprocessing audio files
- `AudioTransformer`: Transformer-based neural network model (4 layers, 4 attention heads, 128 d_model)
- `Trainer`: Training orchestration class with distributed training support, validation, model saving, and TensorBoard logging
- `collate_fn_spectrogram`: Batch collation function that pads/truncates spectrograms to 81 time frames

**predict.py** - Inference script for single audio file predictions using saved model weights

**prepare-data.py** - Dataset download script using torchaudio.datasets.SPEECHCOMMANDS

**example.ipynb** - Comprehensive notebook with both local and Kubeflow distributed training examples, including Kubernetes deployment setup

### Data Pipeline

1. **Audio Loading**: WAV files (16kHz, 1-second clips) from Speech Commands dataset
2. **Preprocessing**: Conversion to mel spectrograms (128 mel bins) using torchaudio transforms
3. **Normalization**: Fixed-length spectrograms (81 time frames) via padding/truncation
4. **Data Splitting**: 95% train, 3% validation, 2% test with reproducible random seed (41)

### Model Architecture

- **Input**: Mel spectrograms (batch_size, 128, 81) 
- **Transformer Encoder**: 4 layers, 4 attention heads, 128 features
- **Output**: 35 classes (word classifications)
- **Aggregation**: Mean pooling over time dimension before final linear layer

### Training Infrastructure

- **Distributed Training**: Full PyTorch DDP (DistributedDataParallel) support for multi-node/multi-GPU training
- **Experiment Tracking**: Automatic timestamped experiment directories in `/data/speech-recognition/runs/`
- **Model Checkpointing**: Best validation accuracy models saved as `.pth` files
- **Monitoring**: TensorBoard integration for loss and accuracy visualization
- **Scheduling**: StepLR scheduler (gamma=0.9, step_size=3)
- **Container Support**: Docker and Kubernetes integration via Kubeflow

### Key Configuration

- **Debug Mode**: Set `debug = True` in train_model.py to use smaller dataset and fewer epochs
- **Batch Size**: 256 (64 in debug mode)
- **Learning Rate**: 0.001 (Adam optimizer, with linear scaling for distributed training)
- **Epochs**: 30 (10 in debug mode)
- **Target Classes**: 35 word commands from Speech Commands v0.02
- **Dataset Path**: `/data/SpeechCommands/speech_commands_v0.02/`

## Dataset

Uses Google's Speech Commands Dataset v0.02 (105,829 audio files, 35 classes). Core commands include "yes", "no", directional words, digits 0-9, and auxiliary words. Data is automatically organized into directories by word label.

## File Structure

- `train_model.py`: Main training pipeline with distributed training support
- `predict.py`: Model inference
- `prepare-data.py`: Dataset download utility
- `example.ipynb`: Development notebook with local and distributed training examples
- `requirements.txt`: Python dependencies (219 packages including PyTorch, torchaudio, Kubeflow)
- `Dockerfile`: Container setup based on PyTorch 2.8.0 + CUDA 12.8
- `kubeflow-runtime-example.yaml`: Kubernetes configuration for distributed training
- `kind-config.yaml`: Local Kubernetes cluster setup
- `/data/speech-recognition/runs/`: Experiment outputs (models, logs)
- `/data/SpeechCommands/`: Dataset location