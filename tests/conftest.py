# tests/conftest.py
import pytest
import yaml
from pathlib import Path

@pytest.fixture
def config():
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def test_model(config):
    from src.model import MultiChannelUNet
    return MultiChannelUNet(
        input_channels=config['model']['in_channels'],
        base_channels=config['model']['base_channels']
    )
