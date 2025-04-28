# src/demo.py

import os
import argparse
import torch
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from model import UNetSeparator
from utils import stft, istft

def process_audio_file(model, input_file, output_file, config):
    """Process a single audio file through the trained model."""
    # Load audio
    print(f"Loading audio file: {input_file}")
    audio, sr = sf.read(input_file)
    
    # Resample if needed
    if sr != config['sample_rate']:
        print(f"Resampling from {sr} to {config['sample_rate']} Hz")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config['sample_rate'])
        sr = config['sample_rate']
    
    # Convert to mono if needed
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        print("Converting stereo to mono")
        audio = np.mean(audio, axis=1)
    
    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio).to(model.device)
    
    # Process in chunks to simulate real-time processing
    chunk_size = sr * 3  # 3 seconds chunks
    hop_size = sr * 1    # 1 second overlap
    
    output_audio = []
    
    # Add padding for the first chunk
    padded_audio = torch.nn.functional.pad(audio_tensor, (chunk_size - hop_size, 0))
    
    # Process audio in chunks with overlap
    for i in tqdm(range(0, len(padded_audio) - chunk_size + hop_size, hop_size)):
        # Extract chunk
        chunk = padded_audio[i:i+chunk_size]
        
        # STFT
        X = stft(
            chunk.unsqueeze(0),
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            win_length=config['win_length']
        )
        
        # Predict mask
        with torch.no_grad():
            X_mag = torch.abs(X).unsqueeze(1)
            mask = model(X_mag)
            
            # Apply mask and reconstruct
            est_mag = mask * X_mag.squeeze(1)
            est_spec = est_mag * torch.exp(1j * torch.angle(X))
            est_chunk = istft(
                est_spec,
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length']
            ).squeeze(0)
        
        # Add to output (only use the second half of each chunk except for the first)
        if i == 0:
            output_audio.append(est_chunk[:-hop_size].cpu().numpy())
        else:
            output_audio.append(est_chunk[hop_size:].cpu().numpy())
    
    # Combine chunks
    output_audio = np.concatenate(output_audio)
    
    # Trim to original length
    output_audio = output_audio[:len(audio)]
    
    # Save output
    print(f"Saving processed audio to: {output_file}")
    sf.write(output_file, output_audio, sr)
    
    return output_file

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UNetSeparator(
        in_channels=config['model']['in_channels'] if 'model' in config and 'in_channels' in config['model'] else 1,
        base_channels=config['model']['base_channels'] if 'model' in config and 'base_channels' in config['model'] else 32
    ).to(device)
    
    # Add device attribute for convenience
    model.device = device
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint (epoch {checkpoint['epoch']+1})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model from checkpoint")
    
    model.eval()
    
    # Process audio file or directory
    if os.path.isfile(args.input):
        # Process single file
        output_file = args.output if args.output else os.path.splitext(args.input)[0] + "_processed.wav"
        process_audio_file(model, args.input, output_file, config)
    elif os.path.isdir(args.input):
        # Process all audio files in directory
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            if filename.endswith(('.wav', '.mp3', '.flac')):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output, os.path.splitext(filename)[0] + "_processed.wav")
                process_audio_file(model, input_path, output_path, config)
    else:
        print(f"Input path does not exist: {args.input}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Sound Separator Demo")
    parser.add_argument('--input', required=True, help='Input audio file or directory')
    parser.add_argument('--output', help='Output audio file or directory')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args)
