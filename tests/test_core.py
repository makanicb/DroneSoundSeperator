# tests/test_core.py
import numpy as np
import torch
import pytest
from pathlib import Path
from src.utils import stft, istft, si_sdr_loss
from src.data_loader import MultiChannelDroneDataset

@pytest.fixture
def test_audio():
    return np.random.randn(16000 * 3, 16).astype(np.float32)  # 3s @16kHz, 16ch

def test_npy_loading(tmp_path):
    # Test .npy file I/O
    test_file = tmp_path / "test.npy"
    audio = np.random.randn(16000, 16).astype(np.float32)
    np.save(test_file, audio)
    loaded = np.load(test_file)
    assert loaded.shape == (16000, 16)
    assert loaded.dtype == np.float32

def test_stft_roundtrip(test_audio):
    # Verify STFT reconstruction
    mono_audio = torch.from_numpy(test_audio[:, 0])
    spec = stft(mono_audio.unsqueeze(0))
    recon = istft(spec)
    assert torch.allclose(mono_audio, recon.squeeze(), atol=1e-3)

def test_dataset_loading(config):
    dataset = MultiChannelDroneDataset(
        data_dir=config['data_dir'],
        sample_rate=config['sample_rate'],
        chunk_size_seconds=3.0
    )
    audio, meta = dataset[0]
    assert audio.shape == (16, config['sample_rate'] * 3)  # (C, S)
    assert isinstance(meta, dict)
