import torch
from src.model import UNetSeparator

def load_model(checkpoint_path, device="cpu"):
    model = UNetSeparator(
        input_channels=16, 
        base_channels=32,
        # Add other required model parameters from your config
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel/normal checkpoints
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    return model.eval().to(device)
