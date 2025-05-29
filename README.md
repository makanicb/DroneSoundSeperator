# DroneSoundSeparator

🚁 A neural network project to isolate drone sounds from noisy environments using 16-channel audio processing and modern audio separation techniques.

## Project Overview

DroneSoundSeparator provides end-to-end waveform processing for multi-channel drone audio separation. The system processes 16-channel WAV files through a complete pipeline: **16-channel WAV → API → STFT → UNet Mask → iSTFT → Cleaned 16-channel WAV**.

**Requirements:**
- CUDA 12.2+ for GPU acceleration
- 16-channel audio input (exact requirement)
- Python 3.11 environment

## Quickstart Guide

### Installation
```bash
# Clone and setup
git clone https://github.com/yourrepo/drone-sound-separation
conda create -n dss python=3.11
conda activate dss
pip install -e ".[dev]"

# Start API
uvicorn src.inference.app:app --reload
```

### API Usage Example
```bash
# Process 16-channel audio file
curl -X POST -F "file_upload=@16channel_recording.wav" http://localhost:8000/separate --output cleaned.wav
```

## Input Requirements

```
-----------------------------------------------
| Parameter       | Requirement               |
-----------------------------------------------
| Channels        | 16 (exact)                |
| Sample Rate     | 44.1kHz or 48kHz          |
| Bit Depth       | 16-bit PCM                |
| Duration        | 1-30 seconds              |
| Max File Size   | 50MB                      |
-----------------------------------------------
```

## Key Improvements

- ✅ **Resumable Training** - Continue from saved checkpoints
- ✅ **Memory-Efficient Training** - Mixed precision + gradient accumulation
- ✅ **Optimized Evaluation** - Reduced GPU memory footprint
- ✅ **Enhanced Reproducibility** - Full RNG state tracking
- ✅ **Dynamic Batch Handling** - Adaptive memory management
- ✅ **16-Channel Processing** - Native multi-channel audio support

## Project Structure

```
DroneSoundSeparator/
├── data/
│   ├── clean_drone_16ch/   # Isolated drone audio (16-channel NPY)
│   ├── noise_16ch/         # Environmental noise audio (16-channel NPY)
│   ├── mixtures/           # Mixed drone+noise samples
│   └── metadata.csv        # Metadata for mixtures
│
├── notebooks/
│   └── explore_data.ipynb  # Data exploration
│
├── src/
│   ├── create_dataset.py   # Script to create mixed audio dataset
│   ├── data_loader.py      # Dataset and DataLoader
│   ├── model.py            # U-Net separation model
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── demo.py             # Real-time processing demo
│   ├── utils.py            # STFT/iSTFT, helpers, metrics
│   └── inference/
│       └── app.py          # FastAPI inference server
│
├── tests/
│   ├── test_data/          # Sample audio files
│   │   ├── 20231001-TEST-44k/   # 44.1kHz test data
│   │   │   ├── audio_chunks/    # Test audio segments
│   │   │   │   ├── chunk_0.npy  # Clean drone audio (16ch)
│   │   │   │   └── chunk_1.npy  # Noise audio (16ch)
│   │   │   └── metadata.json    # Test data metadata
│   ├── conftest.py         # Shared fixtures
│   ├── test_core.py        # I/O and core logic
│   ├── test_model.py       # Model tests
│   ├── test_44khz.py       # 44.1kHz sample rate tests
│   ├── generate_test_data.py # Test data generation script
│   ├── test_perf.py        # Performance tests
│   └── test_edge.py        # Edge cases
│
├── scripts/
│   └── generate_test_audio.py  # Test audio generation
│
├── configs/
│   └── config.yaml         # Experiment configuration
│
├── experiments/
│   └── run1/               # Logs, checkpoints, TensorBoard
│
├── convert_wav_to_npy.py   # Audio format conversion script
├── requirements.txt
├── requirements-test.txt   # Testing dependencies
├── .gitignore
└── README.md
```

## Dependencies

### Core Requirements
- Python 3.11
- PyTorch 2.0.1+cu118
- Torchaudio 2.0.2
- FastAPI 0.95.0
- libsndfile1 (system package)

### Installation
```bash
pip install -r requirements.txt
```

## Setup

1. **Convert audio files:**
   ```bash
   python convert_wav_to_npy.py --src-root /path/to/raw/audio --dest-root data
   ```

2. **Create dataset:**
   ```bash
   python src/create_dataset.py --config configs/config.yaml
   ```

## Training

### Basic Usage
```bash
python src/train.py --config configs/config.yaml
```

### Resume Training
```bash
python src/train.py --config configs/config.yaml --resume experiments/run1/ckpt_epoch10.pt
```

### Key Features
- 40% memory reduction via mixed precision
- Configurable gradient accumulation
- Automatic batch splitting for OOM prevention

## Model Architecture

### U-Net Based Separator
- **Input:** 16-channel magnitude spectrogram
- **Output:** Time-frequency mask [0,1]
- **Loss:** SI-SDR with phase reconstruction
- **Structure:** 5-layer encoder/decoder with skip connections

## Configuration
```yaml
# configs/config.yaml
training:
  use_amp: true                # Enable mixed precision
  gradient_accumulation: 4     # Accumulate over 4 batches
  max_gpu_utilization: 0.8     # Limit VRAM usage
  
checkpoints:
  save_rng_states: true        # For exact reproducibility
  save_frequency: 5            # Save every 5 epochs
```

## Evaluation

### Basic Usage
```bash
python src/evaluate.py --config configs/config.yaml --checkpoint experiments/run1/best_model.pt
```

### Advanced Options
```bash
--output-dir results/       # Save per-sample metrics
--batch-size 8              # Control memory usage
--compute-sdr false         # Disable SDR for faster runs
```

## Visualization

### TensorBoard Integration:
```bash
tensorboard --logdir experiments/run1/logs --bind_all
```

**Track:**
- Loss curves with smoothing
- Validation metrics by SNR
- GPU memory utilization
- Audio waveform comparisons

## Testing

### Generate Test Audio
```bash
python scripts/generate_test_audio.py --channels 16 --duration 5 --output test_16ch.wav
```

### Run Full Suite
```bash
pytest tests/ --verbose --log-level=DEBUG --cov=src --cov-report=html
```

### Key Tests
- Training resumption consistency
- Memory leak detection
- 44.1kHz processing fidelity
- 16-channel validation cases

## Troubleshooting

### Common Issues

**"ValueError: too many values to unpack"**
- Verify input audio has exactly 16 channels

**"400: Invalid channels"**
- Check with `soxi <file>` to validate channel count

**CUDA OOM errors**
- Reduce audio duration below 30 seconds
- Lower batch size in configuration

## Monitoring

### GPU Utilization
```bash
watch -n 1 nvidia-smi
```

### Memory Profiling
```bash
python -m torch.utils.bottleneck src/train.py --config configs/config.yaml
```

## Contributing Guidelines

### Channel Dimension Rules
- All audio processing must maintain 16-channel structure
- Tests must include 16-channel validation cases
- Pre-commit checks for channel count validation

### Development Setup
```bash
pip install -r requirements-test.txt
pre-commit install
```

## Support Resources

- **Email:** audio-support@yourcompany.com
- **Discord:** https://discord.gg/yourcommunity
- **Status Page:** https://status.yourcompany.com

## License & Citation

Copyright 2025. MIT License - See LICENSE for details.

If using this work in research, please cite:
[Pending publication]
