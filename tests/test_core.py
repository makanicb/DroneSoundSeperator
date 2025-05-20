# tests/test_core.py
import numpy as np
import torch
import pytest
from pathlib import Path
from src.utils import stft, istft, si_sdr_loss
from src.data_loader import MultiChannelDroneDataset

@pytest.fixture
def test_audio(config):
    return np.random.randn(config['sample_rate'] * 3, 16).astype(np.float32)  # 3s @44.1kHz, 16ch

def test_npy_loading(tmp_path, config):
    # Test .npy file I/O
    test_file = tmp_path / "test.npy"
    audio = np.random.randn(config['sample_rate'], 16).astype(np.float32)
    np.save(test_file, audio)
    loaded = np.load(test_file)
    assert loaded.shape == (config['sample_rate'], 16)
    assert loaded.dtype == np.float32

def test_stft_roundtrip(test_audio):
    # Verify STFT reconstruction
    mono_audio = torch.from_numpy(test_audio[:, 0])
    spec = stft(mono_audio.unsqueeze(0).unsqueeze(0))
    recon = istft(spec, length=mono_audio.shape[0])
    assert torch.allclose(mono_audio, recon.squeeze(), atol=1e-3)

def test_dataset_loading(config):
    dataset = MultiChannelDroneDataset(
        mixtures_dir=config['mix_dir'],
        clean_dir=config['clean_dir'],
        noise_dir=config['noise_dir'],
        dataset_overview_path=config['dataset_overview'],
        sample_rate=config['sample_rate'],
        chunk_size_seconds=3.0,
        split=1.0
    )
    mixed, clean = dataset[0]  # Changed variable names
    assert mixed.shape == (16, config['sample_rate'] * 3)  # (C, S)
    assert clean.shape == (16, config['sample_rate'] * 3)  # Add this check
