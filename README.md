# Speech Recognition with PyTorch and Kubeflow

A complete example demonstrating **PyTorch Distributed Data Parallel (DDP)** training for speech recognition using Google's Speech Commands dataset. This project showcases both local development and distributed training on Kubernetes using **Kubeflow Trainer**.

## 🎯 Overview

This repository implements a **Transformer-based neural network** for classifying single-word spoken commands (35 classes) from the Speech Commands v0.02 dataset. The main focus is the comprehensive **`example.ipynb`** notebook that walks you through:

- Local training and development
- Container setup with Docker
- Distributed training on Kubernetes using Kubeflow
- Predict with trained model

## 📋 Quick Start

### 1. Local Environment Setup

**Note**: If you encounter torch installation issues, install PyTorch first:

```bash
pip install torch==2.8
pip install -r requirements.txt
```

### 2. Run the Complete Example

File: `example.ipynb`

This notebook contains everything you need, including:

- Data download and preparation
- Local training examples
- Docker container setup
- Kubernetes cluster creation with Kind
- Kubeflow distributed training
- Predict with trained model.

## 🏗️ Architecture

### Model Architecture

- **Input**: Mel spectrograms (128 mel bins, 81 time frames)
- **Model**: Transformer encoder (4 layers, 4 attention heads, 128 d_model)
- **Output**: 35-class classification (Speech Commands)
- **Training**: PyTorch DDP with automatic mixed precision

### Dataset

- **Source**: Google Speech Commands Dataset v0.02
- **Size**: 105,829 audio files (2.3GB)
- **Classes**: 35 words including "yes", "no", digits 0-9, directions, etc.
- **Format**: 1-second WAV files at 16kHz

## 📁 Project Structure

### Core Files

- **`example.ipynb`** - 📓 **Main notebook with complete workflow**
- **`train_model.py`** - 🚂 Standalone training script
- **`predict.py`** - 🔮 Random audio prediction script
- **`prepare-data.py`** - 📥 Dataset download utility

### Infrastructure Files

- **`Dockerfile`** - 🐳 Container setup (PyTorch 2.8.0 + CUDA 12.8)
- **`kind-config.yaml`** - ☸️ Local Kubernetes cluster configuration
- **`kubeflow-runtime-example.yaml`** - 🎛️ Kubeflow runtime definition
- **`requirements.txt`** - 📦 Python dependencies (219 packages)

## 🚀 Usage Examples

### Data Preparation

```bash
python prepare-data.py
```

### Local Training (Single GPU)

```bash
# Run with single GPU
torchrun --nproc-per-node 1 train_model.py

# Run with multiple GPUs
torchrun --nproc-per-node 2 train_model.py
```

### Random Audio Prediction

```bash
python predict.py
```

Sample output:

```
[ 1/10] ✓ File: /data/SpeechCommands/speech_commands_v0.02/left/ae71797c_nohash_0.wav
       True: 'left' | Predicted: 'left' | Confidence: 95.23%

[ 2/10] ✗ File: /data/SpeechCommands/speech_commands_v0.02/yes/ab123cd4_nohash_1.wav
       True: 'yes' | Predicted: 'no' | Confidence: 78.45%
```

## 🐳 Docker & Kubernetes Setup

### Build Docker Image

```bash
docker build -t speech-recognition-image:0.1 .
```

### Create Local Kubernetes Cluster

```bash
# Create Kind cluster with data volume mounting
kind create cluster --name ml --config kind-config.yaml

# Load Docker image to cluster
kind load docker-image speech-recognition-image:0.1 --name ml
```

### Deploy Kubeflow Runtime

```bash
# Install Kubeflow Trainer operator
export VERSION=v2.0.0
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"

# Apply custom runtime
kubectl apply -f kubeflow-runtime-example.yaml
```

## 📊 Distributed Training with Kubeflow

The **`example.ipynb`** notebook demonstrates distributed training:

```python
from kubeflow.trainer import CustomTrainer, TrainerClient

client = TrainerClient()

# Start distributed training job
job_name = client.train(
    trainer=CustomTrainer(
        func=train_model,
        num_nodes=2,  # Multi-node training
        resources_per_node={
            "cpu": 5,
            "memory": "50Gi",
            # "nvidia.com/gpu": 1,  # Uncomment for GPU
        },
    ),
    runtime=torch_runtime,
)
```

## 🔧 Configuration

### Key Parameters

- **Batch Size**: 256 (64 in debug mode)
- **Learning Rate**: 0.001 with linear scaling for distributed training
- **Epochs**: 30 (10 in debug mode)
- **Data Split**: 95% train, 3% validation, 2% test
- **Debug Mode**: Set `debug = True` in scripts for faster iteration

### Data Paths

- **Dataset**: `/data/SpeechCommands/speech_commands_v0.02/`
- **Experiments**: `/data/speech-recognition/runs/exp-{timestamp}/`
- **Models**: Saved as `.pth` files with best validation accuracy

## 📈 Monitoring

### TensorBoard

```bash
tensorboard --logdir=/data/speech-recognition/runs
```

### Kubernetes Logs

```bash
# Get pods
kubectl get pods

# View training logs
kubectl logs <pod-name> -f
```

## 🛠️ Development Workflow

1. **Start with `example.ipynb`** - Complete guided walkthrough
2. **Local development** - Use `train_model.py` for quick iterations
3. **Test predictions** - Run `predict.py` to validate model performance
4. **Scale up** - Deploy to Kubernetes for distributed training

## 📝 Notes

- **Data Volume**: The setup uses `/data` directory mounted across all containers
- **GPU Support**: Works with both CPU and GPU training
- **Reproducibility**: Fixed random seeds (41) for consistent results
- **Production Ready**: Includes model checkpointing, logging, and monitoring

## 🤝 Contributing

This is a complete example project demonstrating PyTorch DDP and Kubeflow integration. Feel free to adapt the patterns for your own speech recognition or distributed training projects.

---

**💡 Tip**: Start with the `example.ipynb` notebook - it contains the complete workflow and explains each step in detail!
