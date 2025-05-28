# src/create_dataset.py (Memory-Optimized)
import os
import random
import numpy as np
import json
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import gc

def mix_multichannel_audio(clean, noise, snr_db, sample_rate=44100):
    """Mix 16-channel arrays with per-channel SNR adjustment"""
    # Validate shapes
    if clean.shape[1] != 16 or noise.shape[1] != 16:
        raise ValueError("Inputs must be 16-channel (samples, 16)")
    
    # Match lengths
    min_len = min(clean.shape[0], noise.shape[0])
    clean = clean[:min_len, :]
    noise = noise[:min_len, :]
    
    # SNR processing
    scaled_noise = np.zeros_like(noise)
    for c in range(16):
        clean_power = np.mean(clean[:, c]**2)
        noise_power = np.mean(noise[:, c]**2)
        k = np.sqrt(clean_power/(10**(snr_db/10)*noise_power)) if noise_power > 0 else 0
        scaled_noise[:, c] = noise[:, c] * k
    
    # Mix and normalize
    mixed = (clean + scaled_noise)
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-8) * 0.9
    return mixed.astype(np.float32)

def create_dataset(config_path):
    """Memory-efficient dataset creation"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config['data']['mixtures_dir'])
    clean_dir = Path(config['data']['clean_dir'])
    noise_dir = Path(config['data']['noise_dir'])
    
    clean_files = list(clean_dir.glob('*.npy'))
    noise_files = list(noise_dir.glob('*.npy'))
    
    session_counter = 0
    global_meta = []
    
    # Process clean files sequentially
    for clean_path in tqdm(clean_files, desc="Main Processing"):
        try:
            clean = np.load(clean_path)
        except Exception as e:
            print(f"Skipped {clean_path.name}: {str(e)}")
            continue
        
        # Process all SNR levels for this clean file
        for snr in config['data']['snr_levels']:
            noise_path = random.choice(noise_files)
            noise = None
            mixed = None
            
            try:
                noise = np.load(noise_path)
                mixed = mix_multichannel_audio(clean, noise, snr)
                
                # Create session folder
                session_counter += 1
                session_id = f"{datetime.now().strftime('%Y%m%d')}_{session_counter:04d}"
                session_dir = data_dir / session_id
                session_dir.mkdir(exist_ok=True)
                
                # Save chunk
                chunk_path = session_dir / 'audio_chunks' / 'chunk_0.npy'
                chunk_path.parent.mkdir(exist_ok=True)
                np.save(chunk_path, mixed)
                
                # Metadata
                metadata = {
                    "session_id": session_id,
                    "duration": mixed.shape[0]/44100,
                    "snr_db": snr,
                    "audio_chunks_timestamps": [{
                        "chunk_index": 0,
                        "start_time": 0.0,
                        "end_time": mixed.shape[0]/44100
                    }]
                }
                with open(session_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                global_meta.append({
                    "session_id": session_id,
                    "clean": str(clean_path),
                    "noise": str(noise_path)
                })
                
            except Exception as e:
                print(f"Skipped {clean_path.name}+{noise_path.name}: {str(e)}")
            finally:
                # Critical memory cleanup
                del noise, mixed
                gc.collect()
        
        # Cleanup after all SNR levels
        del clean
        gc.collect()
    
    # Save global metadata
    with open(data_dir / 'dataset_overview.json', 'w') as f:
        json.dump(global_meta, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    create_dataset(args.config)
