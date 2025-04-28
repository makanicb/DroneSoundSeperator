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

def create_dataset(config_path):
    """Create a dataset of mixed drone and noise audio files."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup directories
    data_dir = os.path.dirname(config['data_dir'])
    clean_dir = os.path.join(data_dir, 'clean_drone')
    noise_dir = os.path.join(data_dir, 'noise')
    mixture_dir = os.path.join(data_dir, 'mixtures')
    
    # Create output directories if they don't exist
    os.makedirs(mixture_dir, exist_ok=True)
    
    # Get list of audio files
    clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(('.wav', '.flac', '.mp3'))]
    noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(('.wav', '.flac', '.mp3'))]
    
    if not clean_files:
        print(f"No audio files found in {clean_dir}")
        return
    
    if not noise_files:
        print(f"No audio files found in {noise_dir}")
        return
    
    # Prepare metadata
    metadata = []
    
    # Create mixtures
    for i, clean_file in enumerate(tqdm(clean_files, desc="Creating mixtures")):
        clean_name = os.path.basename(clean_file).split('.')[0]
        
        for snr in config['snr_levels']:
            # Pick a random noise file
            noise_file = random.choice(noise_files)
            noise_name = os.path.basename(noise_file).split('.')[0]
            
            # Define output filename
            output_name = f"{clean_name}_mixed_with_{noise_name}_snr{snr}.wav"
            output_file = os.path.join(mixture_dir, output_name)
            
            # Mix files
            mix_info = mix_audio_files(
                clean_file, 
                noise_file, 
                snr, 
                output_file,
                sample_rate=config['sample_rate']
            )
            
            metadata.append(mix_info)
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(data_dir, 'metadata.csv'), index=False)
    
    print(f"Created {len(metadata)} mixture files")
    print(f"Metadata saved to {os.path.join(data_dir, 'metadata.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mixtures dataset for drone sound separation")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    create_dataset(args.config)
