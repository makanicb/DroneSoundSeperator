# DroneSoundSeparator

🚁 A neural network project to isolate drone sounds from noisy environments, inspired by modern audio separation techniques.

## Input Specifications
- Accepts multi-channel (16-channel) audio from UMA-16V2 array
- Supports original sample rates up to 44.1kHz (automatically resampled to 16kHz)
- Processes audio chunks of any length (automatically split into 3s segments)

## Project Structure

```
DroneSoundSeparator/
├── data/
│   ├── clean_drone/       # Isolated drone audio
│   ├── noise/             # Environmental noise audio
│   ├── mixtures/          # Mixed drone+noise samples
│   └── metadata.csv       # Metadata for mixtures
│
├── notebooks/
│   └── explore_data.ipynb # Data exploration
│
├── src/
│   ├── create_dataset.py  # Script to create mixed audio dataset
│   ├── data_loader.py     # Dataset and DataLoader
│   ├── model.py           # U-Net separation model
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   ├── demo.py            # Real-time processing demo
│   └── utils.py           # STFT/iSTFT, helpers, metrics
│
├── configs/
│   └── config.yaml        # Experiment configuration
│
├── experiments/
│   └── run1/              # Logs, checkpoints, TensorBoard
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset:
   - Place clean drone recordings in `data/clean_drone/`
   - Place environmental noise recordings in `data/noise/`
   - Run the dataset creation script:
     ```bash
     python src/create_dataset.py --config configs/config.yaml
     ```

3. Train the model:
   ```bash
   python src/train.py --config configs/config.yaml
   ```

4. Evaluate the model:
   ```bash
   python src/evaluate.py --config configs/config.yaml --checkpoint experiments/run1/best_model.pt
   ```

5. Process your own audio:
   ```bash
   python src/demo.py --input your_audio.wav --checkpoint experiments/run1/best_model.pt
   ```

## Model Architecture

The system uses a U-Net style encoder-decoder architecture for predicting time-frequency masks:

- **Input**: Magnitude spectrogram of the mixed audio
- **Output**: Mask in range [0,1] to be applied to the input spectrogram
- **Architecture**: Encoder-decoder with skip connections
- **Optimization**: SI-SDR loss in time domain

## Training Details

- **Loss Function**: Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
- **Optimizer**: Adam with learning rate scheduling
- **Data Augmentation**: Random cropping and padding
- **Validation**: Regular evaluation using SDR, SIR, SAR metrics

## Evaluation Metrics

- **SDR (Signal-to-Distortion Ratio)**: Overall separation quality
- **SIR (Signal-to-Interference Ratio)**: Rejection of unwanted sources
- **SAR (Signal-to-Artifacts Ratio)**: Absence of artifacts

## Visualization

Training progress can be monitored via TensorBoard:

```bash
tensorboard --logdir experiments/run1/logs
```

## Credits

Based on inspiration from recent advances in deep learning for audio separation tasks.

## License

MIT

---
