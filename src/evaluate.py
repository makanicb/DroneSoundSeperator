# src/evaluate.py
import os
import json
import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle

from data_loader import MultiChannelDroneDataset
from model import UNetSeparator
from utils import compute_sdr_sir_sar

def evaluate(config_path, checkpoint_path, output_dir=None, batch_size=None):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Device setup
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation device: {device}")
    
    # Model setup with parameters from config
    model = UNetSeparator(
        input_channels=config['model']['in_channels'],
        base_channels=config['model']['base_channels']
    ).to(device)
    
    # Load checkpoint with DataParallel handling
    try:
        # First try with weights_only=True (default in PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except pickle.UpicklingError as e:
        print(f"Retrying checkpoint load without weights_only restriction")
        # Fallback to weights_only=False for compatibility
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint['model_state_dict']
    if isinstance(model, torch.nn.DataParallel) and not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif not isinstance(model, torch.nn.DataParallel) and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    
    # Evaluation parameters
    batch_size = batch_size or config['evaluation'].get('batch_size', 8)
    use_amp = config['evaluation'].get('use_amp', True)
    
    # Data loader - updated to MultiChannelDroneDataset
    test_set = MultiChannelDroneDataset(
        mixtures_dir=config['data']['mixtures_dir'],
        clean_dir=config['data']['clean_dir'],
        noise_dir=config['data']['noise_dir'],
        dataset_overview_path=config['data']['dataset_overview'],
        sample_rate=config['data']['sample_rate'],
        chunk_size_seconds=config['data']['max_audio_length'],
        mode='test',
        split=config['data']['train_test_split']
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
                batch_results = process_batch(batch, model, device, use_amp)
                results['sdr'].extend(batch_results['sdr'])
                results['sir'].extend(batch_results['sir'])
                results['sar'].extend(batch_results['sar'])
                results['metadata'].extend(batch_results['metadata'])
            except RuntimeError as e:  # Handle OOM
                if 'CUDA out of memory' in str(e):
                    print(f"OOM detected, processing batch in smaller chunks")
                    process_batch_chunked(batch, model, device, results, use_amp)
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

def process_batch(batch, model, device, use_amp):
    noisy, clean, *metadata = batch
    metadata = metadata[0] if metadata else [None]*len(noisy)
    
    noisy = noisy.to(device, non_blocking=True)
    clean = clean.to(device, non_blocking=True)
    
    with torch.cuda.amp.autocast(enabled=use_amp):
        # Model directly outputs estimated waveform
        est_wav = model(noisy)
    
    return calculate_metrics(clean, est_wav, metadata)

def process_batch_chunked(batch, model, device, results, use_amp):
    chunk_size = max(1, batch[0].shape[0] // 2)
    for i in range(0, batch[0].shape[0], chunk_size):
        chunk = [t[i:i+chunk_size] for t in batch]
        chunk_results = process_batch(chunk, model, device, use_amp)
        results['sdr'].extend(chunk_results['sdr'])
        results['sir'].extend(chunk_results['sir'])
        results['sar'].extend(chunk_results['sar'])
        results['metadata'].extend(chunk_results['metadata'])

def calculate_metrics(clean, est_wav, metadata):
    # Keep tensors on device for efficiency
    metrics = {'sdr': [], 'sir': [], 'sar': [], 'metadata': []}
    
    for i in range(clean.shape[0]):
        try:
            # Compute metrics per sample
            metric_dict = compute_sdr_sir_sar(clean[i], est_wav[i])
            metrics['sdr'].append(metric_dict['sdr'].item())
            metrics['sir'].append(metric_dict['sir'].item())
            metrics['sar'].append(metric_dict['sar'].item())
            metrics['metadata'].append(metadata[i] if metadata else None)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Append zeros to maintain alignment
            metrics['sdr'].append(0.0)
            metrics['sir'].append(0.0)
            metrics['sar'].append(0.0)
            metrics['metadata'].append(metadata[i] if metadata else None)
            
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
