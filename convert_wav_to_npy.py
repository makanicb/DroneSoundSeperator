import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

def process_audio(input_path, output_dir, max_duration=3.0):
    """Process individual audio file with error handling"""
    try:
        # Load with original sample rate
        wav, orig_sr = librosa.load(input_path, sr=None, mono=False)
        
        # Resample to 44.1kHz if needed
        if orig_sr != 44100:
            wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=44100)
        
        # Convert to 16 channels
        if len(wav.shape) == 1:  # Mono
            wav = np.repeat(wav[:, np.newaxis], 16, axis=1)
        else:  # Multi-channel
            wav = convert_to_16ch(wav.T)  # Shape: [samples, 16]
        
        # Trim/pad to 3-second chunks (optional)
        target_samples = int(44100 * max_duration)
        if len(wav) > target_samples:
            wav = wav[:target_samples]
        else:
            wav = np.pad(wav, ((0, target_samples - len(wav)), (0, 0)))
        
        # Save as NPY
        output_path = output_dir / (Path(input_path).stem + ".npy")
        np.save(output_path, wav.astype('float32'))
        
    except Exception as e:
        logging.error(f"Failed {input_path}: {str(e)}")

def convert_to_16ch(wav_array):
    """Channel conversion logic"""
    num_channels = wav_array.shape[1]
    if num_channels == 1:
        return np.repeat(wav_array, 16, axis=1)
    elif 2 <= num_channels < 16:
        repeats = 16 // num_channels
        remainder = 16 % num_channels
        return np.hstack([wav_array] * repeats + [wav_array[:, :remainder]])
    else:
        return wav_array[:, :16]

def convert_dataset(src_root, dest_root):
    """Main conversion function"""
    Path(dest_root).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename='conversion_errors.log', level=logging.ERROR)
    
    categories = {
        "yes_drone": "clean_drone_16ch",
        "unknown": "noise_16ch"
    }
    
    for src_category, dest_category in categories.items():
        input_dir = Path(src_root) / src_category
        output_dir = Path(dest_root) / dest_category
        output_dir.mkdir(exist_ok=True)
        
        print(f"Processing {src_category}...")
        for wav_file in tqdm(list(input_dir.glob("*.wav"))):
            process_audio(wav_file, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert WAV files to 16-channel NPY format')
    parser.add_argument('--src-root', type=str, required=True,
                        help='Path to root directory containing yes_drone/unknown folders')
    parser.add_argument('--dest-root', type=str, default='data',
                        help='Output root directory (default: data)')
    
    args = parser.parse_args()
    
    convert_dataset(
        src_root=args.src_root,
        dest_root=args.dest_root
    )
