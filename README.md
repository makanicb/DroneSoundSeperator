# DroneSoundSeparator

🚁 A neural network project to isolate drone sounds from noisy environments, inspired by modern audio separation techniques.

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
│   ├── data_loader.py     # Dataset and DataLoader
│   ├── model.py           # U-Net separation model
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation metrics
│   └── utils.py           # STFT/iSTFT, helpers
│
├── configs/
│   └── config.yaml        # Experiment configuration
│
├── experiments/
│   └── run1/              # Logs, checkpoints
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

2. Organize your datasets under `data/`.

3. Train the model:
   ```bash
   python src/train.py --config configs/config.yaml
   ```

## Model

Uses a U-Net style encoder-decoder for predicting time-frequency masks on STFT spectrograms.

## Evaluation

Metrics:
- SDR (Signal-to-Distortion Ratio)
- SIR (Signal-to-Interference Ratio)
- SAR (Signal-to-Artifacts Ratio)

## Credits

Based on inspiration from recent advances in deep learning for audio separation tasks.

---
