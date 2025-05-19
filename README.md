DroneSoundSeparator

🚁 A neural network project to isolate drone sounds from noisy environments, inspired by modern audio separation techniques.
Key Improvements

    ✅ Resumable Training - Continue from saved checkpoints

    ✅ Memory-Efficient Training - Mixed precision + gradient accumulation

    ✅ Optimized Evaluation - Reduced GPU memory footprint

    ✅ Enhanced Reproducibility - Full RNG state tracking

    ✅ Dynamic Batch Handling - Adaptive memory management

Input Specifications

    Accepts multi-channel (16-channel) audio from UMA-16V2 array

    Natively supports 44.1kHz sample rate (no resampling needed)

    Processes audio chunks of any length (automatically split into 3s segments)

Project Structure

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
│   └── utils.py            # STFT/iSTFT, helpers, metrics
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

Setup

    Install dependencies:
    bash

pip install -r requirements.txt

Convert audio files:
bash

python convert_wav_to_npy.py --src-root /path/to/raw/audio --dest-root data

Create dataset:
bash

    python src/create_dataset.py --config configs/config.yaml

Training
Basic Usage
bash

python src/train.py --config configs/config.yaml

Resume Training
bash

python src/train.py --config configs/config.yaml --resume experiments/run1/ckpt_epoch10.pt

Key Features

    40% memory reduction via mixed precision

    Configurable gradient accumulation

    Automatic batch splitting for OOM prevention

Model Architecture

U-Net Based Separator

    Input: 16-channel magnitude spectrogram

    Output: Time-frequency mask [0,1]

    Loss: SI-SDR with phase reconstruction

    5-layer encoder/decoder with skip connections

Configuration
yaml

# configs/config.yaml
training:
  use_amp: true                # Enable mixed precision
  gradient_accumulation: 4     # Accumulate over 4 batches
  max_gpu_utilization: 0.8     # Limit VRAM usage
  
checkpoints:
  save_rng_states: true        # For exact reproducibility
  save_frequency: 5            # Save every 5 epochs

Evaluation
Basic Usage
bash

python src/evaluate.py --config configs/config.yaml --checkpoint experiments/run1/best_model.pt

Advanced Options
bash

--output-dir results/       # Save per-sample metrics
--batch-size 8              # Control memory usage
--compute-sdr false         # Disable SDR for faster runs

Visualization

TensorBoard Integration:
bash

tensorboard --logdir experiments/run1/logs --bind_all

Track:

    Loss curves with smoothing

    Validation metrics by SNR

    GPU memory utilization

    Audio waveform comparisons

Testing

Run Full Suite:
bash

pytest tests/ -v --cov=src --cov-report=html

Key Tests:

    Training resumption consistency

    Memory leak detection

    44.1kHz processing fidelity

Monitoring
GPU Utilization
bash

watch -n 1 nvidia-smi

Memory Profiling
bash

python -m torch.utils.bottleneck src/train.py --config configs/config.yaml

License

MIT License - See LICENSE for details
