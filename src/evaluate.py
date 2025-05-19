# src/evaluate.py
import os
import json
import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_loader import DroneMixtureDataset
from model import UNetSeparator
from utils import stft, istft, compute_sdr_sir_sar

def evaluate(config_path, checkpoint_path, output_dir=None, batch_size=None):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Device setup
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation device: {device}")
    
    # Model setup
    model = UNetSeparator().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval()
    
    # Evaluation parameters
    batch_size = batch_size or config['evaluation'].get('batch_size', 8)
    use_amp = config['evaluation'].get('use_amp', True)
    
    # Data loader
    test_set = DroneMixtureDataset(
        data_dir=config['data_dir'],
        snr_levels=config['snr_levels'],
        sample_rate=config['sample_rate'],
        mode='test'
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Results storage
    results = {
        'sdr': [], 'sir': [], 'sar': [],
        'metadata': [],
        'config': {
            'batch_size': batch_size,
            'use_amp': use_amp,
            'checkpoint': checkpoint_path
        }
    }
    
    # Evaluation loop
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating"):
            try:
                batch_results = process_batch(batch, model, device, config, use_amp)
                results['sdr'].extend(batch_results['sdr'])
                results['sir'].extend(batch_results['sir'])
                results['sar'].extend(batch_results['sar'])
                results['metadata'].extend(batch_results['metadata'])
            except RuntimeError as e:  # Handle OOM
                if 'CUDA out of memory' in str(e):
                    print(f"OOM detected, processing batch in smaller chunks")
                    process_batch_chunked(batch, model, device, config, results, use_amp)
                else:
                    raise e
                    
            torch.cuda.empty_cache()
    
    # Save results
    output_dir = output_dir or os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "evaluation_results.json")
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nEvaluation completed")
    print(f"Average SDR: {np.mean(results['sdr']):.2f} dB")
    print(f"Average SIR: {np.mean(results['sir']):.2f} dB")
    print(f"Average SAR: {np.mean(results['sar']):.2f} dB")
    print(f"Full results saved to {result_path}")
    
    return results

def process_batch(batch, model, device, config, use_amp):
    noisy, clean, *metadata = batch
    metadata = metadata[0] if metadata else [None]*len(noisy)
    
    noisy = noisy.to(device, non_blocking=True)
    clean = clean.to(device, non_blocking=True)
    
    with torch.cuda.amp.autocast(enabled=use_amp):
        X = stft(noisy, 
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length'])
        mask = model(torch.abs(X))
        est_wav = reconstruct_audio(X, mask, config)
        
    return calculate_metrics(clean, est_wav, metadata)

def process_batch_chunked(batch, model, device, config, results, use_amp):
    chunk_size = max(1, batch[0].shape[0] // 2)
    for i in range(0, batch[0].shape[0], chunk_size):
        chunk = [t[i:i+chunk_size] for t in batch]
        chunk_results = process_batch(chunk, model, device, config, use_amp)
        results['sdr'].extend(chunk_results['sdr'])
        results['sir'].extend(chunk_results['sir'])
        results['sar'].extend(chunk_results['sar'])
        results['metadata'].extend(chunk_results['metadata'])

def calculate_metrics(clean, est_wav, metadata):
    clean_np = clean.cpu().numpy()
    est_np = est_wav.float().cpu().numpy()
    
    metrics = {'sdr': [], 'sir': [], 'sar': [], 'metadata': []}
    for i in range(clean_np.shape[0]):
        try:
            sdr, sir, sar = compute_sdr_sir_sar(clean_np[i], est_np[i])
            metrics['sdr'].append(sdr)
            metrics['sir'].append(sir)
            metrics['sar'].append(sar)
            metrics['metadata'].append(metadata[i] if metadata else None)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint, args.output_dir, args.batch_size)
