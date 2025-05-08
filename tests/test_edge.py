# tests/test_edge.py
import numpy as np
import torch
import pytest

def test_silent_input(test_model):
    # Silent input should produce near-zero output - make sure channels match model input_channels
    model_input_channels = test_model.input_channels
    silent = torch.zeros(1, model_input_channels, 256, 256)
    mask = test_model(silent)
    assert torch.max(mask) < 0.01

def test_clipped_audio():
    # Test clipping handling
    clipped = np.random.uniform(-1.5, 1.5, (16000, 16)).astype(np.float32)
    assert np.max(np.abs(clipped)) > 1.0  # Verify test is valid
    dataset = MultiChannelDroneDataset(...)
    processed, _ = dataset.process_audio(clipped)
    assert np.max(np.abs(processed)) <= 1.0  # Should be normalized
