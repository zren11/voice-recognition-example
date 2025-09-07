# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based voice recognition system that trains a Transformer model on the Speech Commands dataset v0.02. The project implements audio classification for single-word spoken commands using mel spectrograms as input features.

## Development Commands

**Training:**
```bash
python train.py
```

**Prediction/Inference:**
```bash
python predict.py
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**TensorBoard Visualization:**
```bash
tensorboard --logdir=runs
```

## Architecture

### Core Components

**train.py** - Main training script containing:
- `SpeechCommandsDataset`: Custom PyTorch Dataset for loading and preprocessing audio files
- `AudioTransformer`: Transformer-based neural network model (4 layers, 4 attention heads, 128 d_model)
- `Trainer`: Training orchestration class with validation, model saving, and TensorBoard logging
- `collate_fn_spectrogram`: Batch collation function that pads/truncates spectrograms to 81 time frames

**predict.py** - Inference script for single audio file predictions using saved model weights

**example.ipynb** - Jupyter notebook with exploratory code and dataset download functionality

### Data Pipeline

1. **Audio Loading**: WAV files (16kHz, 1-second clips) from Speech Commands dataset
2. **Preprocessing**: Conversion to mel spectrograms (128 mel bins) using torchaudio transforms
3. **Normalization**: Fixed-length spectrograms (81 time frames) via padding/truncation
4. **Data Splitting**: 70% train, 20% validation, 10% test with reproducible random seed

### Model Architecture

- **Input**: Mel spectrograms (batch_size, 128, 81) 
- **Transformer Encoder**: 4 layers, 4 attention heads, 128 features
- **Output**: 35 classes (word classifications)
- **Aggregation**: Mean pooling over time dimension before final linear layer

### Training Infrastructure

- **Experiment Tracking**: Automatic timestamped experiment directories in `runs/`
- **Model Checkpointing**: Best validation accuracy models saved as `.pth` files
- **Monitoring**: TensorBoard integration for loss and accuracy visualization
- **Scheduling**: StepLR scheduler (gamma=0.5, step_size=3)

### Key Configuration

- **Debug Mode**: Set `debug = True` in train.py to use only 10k samples and 2 epochs
- **Batch Size**: 512
- **Learning Rate**: 0.001 (Adam optimizer)
- **Target Classes**: 35 word commands from Speech Commands v0.02

## Dataset

Uses Google's Speech Commands Dataset v0.02 (105,829 audio files, 35 classes). Core commands include "yes", "no", directional words, digits 0-9, and auxiliary words. Data is automatically organized into directories by word label.

## File Structure

- `train.py`: Main training pipeline
- `predict.py`: Model inference
- `example.ipynb`: Development notebook
- `requirements.txt`: Python dependencies
- `runs/`: Experiment outputs (models, logs)
- `SpeechCommands/`: Dataset location
- `speech_commands_v0.02.tar.gz`: Dataset archive