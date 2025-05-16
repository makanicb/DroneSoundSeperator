# DroneSoundSeparator

🚁 A neural network project to isolate drone sounds from noisy environments, inspired by modern audio separation techniques.

## Input Specifications
- Accepts multi-channel (16-channel) audio from UMA-16V2 array
- Natively supports 44.1kHz sample rate (no resampling needed)
- Processes audio chunks of any length (automatically split into 3s segments)

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
```

## Setup

1. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Convert WAV files to NPY format:
   ```bash
   python convert_wav_to_npy.py --src-root /path/to/raw/audio --dest-root data
   ```
   This script:
   - Converts WAV files from "yes_drone" folder to NPY files in "clean_drone_16ch"
   - Converts WAV files from "unknown" folder to NPY files in "noise_16ch" 
   - Handles channel conversion to 16-channel format
   - Maintains native 44.1kHz sample rate

3. Prepare your dataset:
   - Ensure clean drone recordings are in `data/clean_drone_16ch/`
   - Ensure environmental noise recordings are in `data/noise_16ch/`
   - Run the dataset creation script:
     ```bash
     python src/create_dataset.py --config configs/config.yaml
     ```

4. Train the model:
   ```bash
   python src/train.py --config configs/config.yaml
   ```

5. Evaluate the model:
   ```bash
   python src/evaluate.py --config configs/config.yaml --checkpoint experiments/run1/best_model.pt
   ```

6. Process your own audio:
   ```bash
   python src/demo.py --input your_audio.wav --checkpoint experiments/run1/best_model.pt
   ```

## Model Architecture

The system uses a U-Net style encoder-decoder architecture for predicting time-frequency masks:

- **Input**: Magnitude spectrogram of the mixed audio (44.1kHz native)
- **Output**: Mask in range [0,1] to be applied to the input spectrogram
- **Architecture**: Encoder-decoder with skip connections
- **Optimization**: SI-SDR loss in time domain

## Training Details

- **Loss Function**: Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
- **Optimizer**: Adam with learning rate scheduling
- **Data Augmentation**: Random cropping and padding
- **Validation**: Regular evaluation using SDR, SIR, SAR metrics

## STFT Parameters for 44.1kHz Audio

- **Sample Rate**: 44.1kHz (native)
- **FFT Size**: 2048 samples
- **Hop Length**: 441 samples
- **Window Length**: 2048 samples

## Evaluation Metrics

- **SDR (Signal-to-Distortion Ratio)**: Overall separation quality
- **SIR (Signal-to-Interference Ratio)**: Rejection of unwanted sources
- **SAR (Signal-to-Artifacts Ratio)**: Absence of artifacts

## Visualization

Training progress can be monitored via TensorBoard:

```bash
tensorboard --logdir experiments/run1/logs
```

## 🧪 Testing Suite

We use `pytest` for comprehensive unit testing. To run all tests:

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

To generate test data for 44.1kHz testing:

```bash
python tests/generate_test_data.py
```

## Credits

Based on inspiration from recent advances in deep learning for audio separation tasks.

## License

MIT

---