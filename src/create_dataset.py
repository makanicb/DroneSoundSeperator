# src/create_dataset.py
import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import yaml
import argparse

def mix_audio_files(clean_file, noise_file, snr_db, output_file, sample_rate=16000):
    """Mix clean drone sound with noise at a specific SNR level."""
    # Load files
    clean, _ = librosa.load(clean_file, sr=sample_rate, mono=True)
    noise, _ = librosa.load(noise_file, sr=sample_rate, mono=True)
    
    # Match lengths
    if len(clean) > len(noise):
        # Repeat noise if needed
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
        noise = noise[:len(clean)]
    else:
        # Truncate noise if needed
        start = random.randint(0, len(noise) - len(clean))
        noise = noise[start:start + len(clean)]
    
    # Calculate scaling factor for desired SNR
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate scaling factor for noise
    k = np.sqrt(clean_power / (10 ** (snr_db / 10) * noise_power))
    
    # Mix with scaling
    scaled_noise = k * noise
    mixture = clean + scaled_noise
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixture))
    if max_val > 1.0:
        mixture = mixture / max_val * 0.9
    
    # Save the mixture
    sf.write(output_file, mixture, sample_rate)
    
    return {
        'mixture': output_file,
        'clean': clean_file,
        'noise': noise_file,
        'snr': snr_db
    }

def mix_audio_npy(clean_path, noise_path, snr_db, output_path):
    """Mix mono drone audio with multichannel noise at target SNR"""
    clean = np.load(clean_path)[:, 0]  # Ensure mono (S,) shape
    noise = np.load(noise_path)  # (S, C)
    
    # Match lengths
    min_len = min(len(clean), noise.shape[0])
    clean = clean[:min_len]
    noise = noise[:min_len, :]
    
    # SNR scaling per channel
    clean_power = np.mean(clean**2)
    scaled_noise = np.zeros_like(noise)
    for c in range(noise.shape[1]):
        noise_power = np.mean(noise[:, c]**2)
        k = np.sqrt(clean_power / (10**(snr_db/10) * noise_power))
        scaled_noise[:, c] = noise[:, c] * k
    
    # Mix and normalize
    mixed = clean.reshape(-1, 1) + scaled_noise  # Broadcast to multichannel
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-8) * 0.9  # Peak at -1dBFS
    
    np.save(output_path, mixed.astype(np.float32))
    return output_path

def create_dataset(config_path):
    """Create a dataset of mixed drone and noise audio files."""

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    metadata = []
    clean_files = [f for f in os.listdir(os.path.join(config['data_dir'], 'clean_drone') 
                  if f.endswith('.npy')]
    noise_files = [f for f in os.listdir(os.path.join(config['data_dir'], 'noise')) 
                   if f.endswith('.npy')]

    for clean_file in tqdm(clean_files, desc='Creating mixtures'):
        clean_path = os.path.join(config['data_dir'], 'clean_drone', clean_file)
        for snr in config['snr_levels']:
            noise_file = random.choice(noise_files)
            noise_path = os.path.join(config['data_dir'], 'noise', noise_file)
            
            output_name = f"{os.path.splitext(clean_file)[0]}_snr{snr}.npy"
            output_path = os.path.join(config['data_dir'], 'mixtures', output_name)
            
            mix_audio_npy(clean_path, noise_path, snr, output_path)
            
            metadata.append({
                'mixture': output_path,
                'clean': clean_path,
                'noise': noise_path,
                'snr': snr
            })
    
    # Save metadata
    with open(os.path.join(config['data_dir'], 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created {len(metadata)} mixture files")
    print(f"Metadata saved to {os.path.join(data_dir, 'metadata.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mixtures dataset for drone sound separation")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    create_dataset(args.config)
