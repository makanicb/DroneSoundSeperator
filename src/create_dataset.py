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

def mix_audio_npy(clean_path, noise_path, snr_db, sample_rate=44100):
    """Mix mono drone audio with multichannel noise at target SNR. Returns mixed array."""
    clean = np.load(clean_path)[:, 0]  # Ensure mono (S,) shape
    noise = np.load(noise_path)        # (S, C)
    
    # Match lengths
    min_len = min(len(clean), noise.shape[0])
    clean = clean[:min_len]
    noise = noise[:min_len, :]
    
    # SNR scaling per channel
    clean_power = np.mean(clean**2)
    scaled_noise = np.zeros_like(noise)
    for c in range(noise.shape[1]):
        noise_power = np.mean(noise[:, c]**2)
        k = np.sqrt(clean_power / (10 ** (snr_db / 10) * noise_power))
        scaled_noise[:, c] = noise[:, c] * k
    
    # Mix and normalize
    mixed = clean.reshape(-1, 1) + scaled_noise  # Broadcast to multichannel
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-8) * 0.9  # Peak at -1dBFS
    
    return mixed.astype(np.float32)

def create_dataset(config_path):
    """Create dataset organized into session folders compatible with data_loader.py."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config['data_dir'])
    sample_rate = config.get('sample_rate', 44100)
    
    # Load clean and noise files
    clean_dir = data_dir / 'clean_drone'
    noise_dir = data_dir / 'noise'
    
    clean_files = list(clean_dir.glob('*.npy'))
    noise_files = list(noise_dir.glob('*.npy'))
    
    if not clean_files or not noise_files:
        raise FileNotFoundError("Missing clean/noise .npy files.")
    
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
                mixed = mix_audio_npy(clean_path, noise_path, snr, sample_rate)
                np.save(chunk_path, mixed)
            except Exception as e:
                print(f"Error processing {clean_path.name}: {e}")
                continue
            
            # Calculate duration
            duration = mixed.shape[0] / sample_rate
            
            # Generate per-session metadata
            metadata = {
                "session_id": session_id,
                "clean_source": str(clean_path),
                "noise_source": str(noise_path),
                "snr_db": snr,
                "sample_rate": sample_rate,
                "audio_chunks_timestamps": [{
                    "chunk_index": 0,
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration
                }]
            }
            
            # Save session metadata
            with open(session_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            global_meta.append({
                "session_id": session_id,
                "mixture_path": str(chunk_path)
            })
    
    # Save global metadata (optional)
    with open(data_dir / 'dataset_overview.json', 'w') as f:
        json.dump(global_meta, f, indent=2)
    
    print(f"Generated {session_counter} sessions in {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset organized into session folders.")
    parser.add_argument('--config', default='config.yaml', help="Path to config file")
    args = parser.parse_args()
    create_dataset(args.config)
