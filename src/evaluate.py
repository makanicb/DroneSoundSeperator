# src/evaluate.py
import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_loader import DroneMixtureDataset
from model import UNetSeparator
from utils import stft, istft, compute_sdr_sir_sar

def evaluate(config_path, checkpoint_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset and loader
    test_dataset = DroneMixtureDataset(
        data_dir=config['data_dir'],
        snr_levels=config['snr_levels'],
        sample_rate=config['sample_rate'],
        mode='test'  # Assuming you add a mode parameter to the dataset class
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Initialize model and load checkpoint
    model = UNetSeparator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Evaluation metrics
    all_sdr, all_sir, all_sar = [], [], []

    # Evaluation metrics with metadata tracking
    results = {                                
        'sdr': [],                             
        'sir': [],                             
        'sar': [],                             
        'metadata': []                         
    }
    
    # Process test set
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):

            if len(batch) == 2:  # (audio, metadata) format  <-- ADDED (line ~18)
                noisy_wav, clean_wav, *maybe_metadata = batch
                batch_metadata = maybe_metadata[0] if maybe_metadata else None
            else:                                               
                noisy_wav, clean_wav = batch                    
                batch_metadata = None

            noisy_wav = noisy_wav.to(device)
            clean_wav = clean_wav.to(device)
            
            # STFT of noisy input
            X = stft(
                noisy_wav, 
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length']
            )
            
            # Predict mask
            mask = model(torch.abs(X))
            
            # Apply mask and reconstruct
            est_mag = mask * torch.abs(X)
            est_spec = est_mag * torch.exp(1j * torch.angle(X))
            est_wav = istft(
                est_spec,
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length']
            )
            
            # Move to CPU for evaluation
            clean_np = clean_wav.cpu().numpy()
            est_np = est_wav.cpu().numpy()
            
            # Compute metrics for each sample in batch
            for i in range(clean_np.shape[0]):
                sdr, sir, sar = compute_sdr_sir_sar(clean_np[i], est_np[i])
                results['sdr'].append(sdr)                     
                results['sir'].append(sir)                     
                results['sar'].append(sar)                     
                if batch_metadata:                             
                    results['metadata'].append(batch_metadata[i])

    # Save full results with metadata
    result_path = os.path.join(os.path.dirname(checkpoint_path), "evaluation_results.json")  
    with open(result_path, 'w') as f:  
        json.dump(results, f, indent=2) 
    
    # Compute average metrics
    avg_sdr = np.mean(results['sdr'])
    avg_sir = np.mean(results['sir'])
    avg_sar = np.mean(results['sar'])
    
    print(f"Evaluation Results:")
    print(f"SDR: {avg_sdr:.2f} dB")
    print(f"SIR: {avg_sir:.2f} dB")
    print(f"SAR: {avg_sar:.2f} dB")
    
    return {
        'sdr': avg_sdr,
        'sir': avg_sir,
        'sar': avg_sar
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint)
