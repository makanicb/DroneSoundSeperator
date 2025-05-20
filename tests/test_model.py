# tests/test_model.py
import torch
import pytest
from src.model import MultiChannelUNet, UNetSeparator
from src.utils import si_sdr_loss, stft, istft

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

def test_unet_separator_shapes():
    """Test that UNetSeparator maintains correct input/output shapes"""
    model = UNetSeparator(n_fft=2048, hop_length=441, input_channels=16)
    dummy_input = torch.randn(2, 16, 44100 * 3)  # Batch of 2, 16ch, 3s @44.1kHz
    output = model(dummy_input)
    assert output.shape == dummy_input.shape  # Shape preservation check

def test_stft_consistency():
    """Verify STFT reconstruction accuracy with utils.py functions"""
    wav = torch.randn(1, 16, 44100)  # Single example, 16ch, 1s @44.1kHz
    complex_spec = stft(wav, n_fft=2048, hop_length=441)
    reconstructed = istft(complex_spec, n_fft=2048, hop_length=441, length=44100)
    assert torch.allclose(wav, reconstructed, atol=1e-3)  # ~0.1% tolerance
