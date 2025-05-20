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
    clipped = np.random.uniform(-1.5, 1.5, (config['sample_rate'], 16)).astype(np.float32)
    assert np.max(np.abs(clipped)) > 1.0  # Verify test is valid
    dataset = MultiChannelDroneDataset(
        mixtures_dir=config['mix_dir'],
        clean_dir=config['clean_dir'],
        noise_dir=config['noise_dir'],
        dataset_overview_path=config['dataset_overview']
    )
    processed, _ = dataset.process_audio(clipped)

    # Convert to numpy if it's a tensor
    if isinstance(processed, torch.Tensor):
        processed = processed.numpy()

    assert np.max(np.abs(processed)) <= 1.0  # Should be normalized
