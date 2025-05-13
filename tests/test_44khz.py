# tests/test_44khz.py
import pytest
import numpy as np
import torch
from pathlib import Path
from src.utils import stft, istft

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
    assert spec.shape == (1, 16, 1025, 300)

def test_config_validation(config):
    # Ensure config matches 44.1kHz requirements
    assert config['sample_rate'] == 44100
    assert config['n_fft'] == 2048
    assert config['hop_length'] == 441

def test_44khz_dataset_chunks():
    # Verify dataset handles 44.1kHz chunks
    dataset = MultiChannelDroneDataset(
        data_dir="tests/test_data",
        sample_rate=44100,
        chunk_size_seconds=3.0
    )
    audio, _ = dataset[0]
    assert audio.shape == (16, 44100 * 3)  # (C, S)
