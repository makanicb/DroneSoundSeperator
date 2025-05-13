# tests/test_model.py
import torch
import pytest
from src.model import MultiChannelUNet
from src.utils import si_sdr_loss

@pytest.fixture
def test_model():
    return MultiChannelUNet(input_channels=16, output_channels=16)

def test_mask_output(test_model):
    # Check mask values are in [0,1]
    dummy_input = torch.randn(1, 16, 256, 256)  # (B,C,F,T)
    mask = test_model(dummy_input)
    assert torch.all(mask >= 0) and torch.all(mask <= 1)

def test_model_shapes(test_model):
    # Verify input/output dimensions
    spec = torch.randn(1, 16, 1025, 256)  # Simulated spectrogram
    out = test_model(spec)
    assert out.shape == spec.shape  # Same dims as input

def test_sisdr_loss(config):
    # Test loss computation
    target = torch.randn(1, config['sample_rate'])
    est = target + 0.1 * torch.randn_like(target)
    loss = si_sdr_loss(est, target)
    assert loss.item() < 0  # SI-SDR is negative when imperfect
