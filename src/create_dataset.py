# src/create_dataset.py
import os
import random
import numpy as np
import json
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

def mix_multichannel_audio(clean_path, noise_path, snr_db, sample_rate=44100):
    """Mix 16-channel clean with 16-channel noise at target SNR (per-channel processing)."""
    # Load both as 16-channel arrays
    clean = np.load(clean_path)  # Shape: (samples, 16)
    noise = np.load(noise_path)  # Shape: (samples, 16)
    
    # Validate channel count
    if clean.shape[1] != 16 or noise.shape[1] != 16:
        raise ValueError("Both clean and noise must be 16-channel (samples, 16)")
    
    # Match lengths
    min_len = min(clean.shape[0], noise.shape[0])
    clean = clean[:min_len, :]
    noise = noise[:min_len, :]
    
    # Calculate scaling factors per channel
    scaled_noise = np.zeros_like(noise)
    for c in range(16):
        clean_power = np.mean(clean[:, c]**2)
        noise_power = np.mean(noise[:, c]**2)
        
        # Handle zero-power cases
        if noise_power == 0:
            k = 0
        else:
            k = np.sqrt(clean_power / (10 ** (snr_db / 10) * noise_power))
        
        scaled_noise[:, c] = noise[:, c] * k
    
    # Mix and normalize
    mixed = clean + scaled_noise
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-8) * 0.9  # Peak at -1dBFS
    
    return mixed.astype(np.float32)

def create_dataset(config_path):
    """Create dataset with 16-channel clean/noise inputs and session folders."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config['data_dir'])
    sample_rate = config.get('sample_rate', 44100)
    
    # Input directories (both expected to contain 16-channel .npy files)
    clean_dir = data_dir / 'clean_drone_16ch'
    noise_dir = data_dir / 'noise_16ch'
    
    clean_files = list(clean_dir.glob('*.npy'))
    noise_files = list(noise_dir.glob('*.npy'))
    
    if not clean_files or not noise_files:
        raise FileNotFoundError("Missing 16-channel .npy files in clean/noise directories")
    
    session_counter = 0
    global_meta = []
    
    for clean_path in tqdm(clean_files, desc="Processing"):
        for snr in config['snr_levels']:
            noise_path = random.choice(noise_files)
            
            # Create session folder
            session_counter += 1
            session_id = f"{datetime.now().strftime('%Y%m%d')}_{session_counter:04d}"
            session_dir = data_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            # Create audio_chunks and save mixture
            audio_chunks_dir = session_dir / 'audio_chunks'
            audio_chunks_dir.mkdir(exist_ok=True)
            chunk_path = audio_chunks_dir / 'chunk_0.npy'
            
            try:
                mixed = mix_multichannel_audio(clean_path, noise_path, snr, sample_rate)
                np.save(chunk_path, mixed)
            except Exception as e:
                print(f"Skipped {clean_path.name}: {str(e)}")
                continue
            
            # Metadata generation
            duration = mixed.shape[0] / sample_rate
            metadata = {
                "session_id": session_id,
                "channels": 16,
                "snr_db": snr,
                "audio_chunks_timestamps": [{
                    "chunk_index": 0,
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration
                }]
            }
            
            with open(session_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            global_meta.append({
                "session_id": session_id,
                "clean": str(clean_path),
                "noise": str(noise_path)
            })
    
    # Save global metadata
    with open(data_dir / 'dataset_overview.json', 'w') as f:
        json.dump(global_meta, f, indent=2)
    
    print(f"Created {session_counter} sessions with 16-channel mixtures")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 16-channel drone dataset")
    parser.add_argument('--config', required=True, help="Path to config.yaml")
    args = parser.parse_args()
    create_dataset(args.config)
