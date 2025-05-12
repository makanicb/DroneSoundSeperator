# tests/test_edge.py
import numpy as np
import torch
import pytest
from src.data_loader import MultiChannelDroneDataset

def test_silent_input(test_model):
    # Verify output shape instead of value for untrained model
    model_input_channels = test_model.input_channels
    silent = torch.zeros(1, model_input_channels, 256, 256)
    mask = test_model(silent)
    assert mask.shape == (1, 16, 256, 256)  # Example check for output shape

def test_clipped_audio(config):
    # Test clipping handling
    clipped = np.random.uniform(-1.5, 1.5, (16000, 16)).astype(np.float32)
    assert np.max(np.abs(clipped)) > 1.0  # Verify test is valid
    dataset = MultiChannelDroneDataset(data_dir=config['data_dir'])
    processed, _ = dataset.process_audio(clipped)
    assert np.max(np.abs(processed)) <= 1.0  # Should be normalized
