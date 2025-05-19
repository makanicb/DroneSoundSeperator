# tests/test_44khz.py
import pytest
import numpy as np
import torch
from pathlib import Path
from src.utils import stft, istft
from src.data_loader import MultiChannelDroneDataset

def test_44khz_stft_dims():
    # Verify STFT output dimensions for 44.1kHz
    audio_3s = torch.randn(1, 16, 44100 * 3)  # 3s @44.1kHz
    spec = stft(
        audio_3s,
        n_fft=2048,
        hop_length=441,
        win_length=2048
    )
    # Expected: (B, C, Freq=1025, Time=300)
    assert spec.shape == (1, 16, 1025, 301)

def test_config_validation(config):
    # Ensure config matches 44.1kHz requirements
    assert config['sample_rate'] == 44100
    assert config['n_fft'] == 2048
    assert config['hop_length'] == 441

def test_44khz_dataset_chunks(config):
    # Verify dataset handles 44.1kHz chunks
    dataset = MultiChannelDroneDataset(
        mixtures_dir=config['mix_dir'],
        clean_dir=config['clean_dir'],
        noise_dir=config['noise_dir'],
        dataset_overview_path=config['dataset_overview'],
        sample_rate=44100,
        chunk_size_seconds=3.0,
        split=1.0
    )
    audio, _ = dataset[0]
    assert audio.shape == (16, 44100 * 3)  # (C, S)
