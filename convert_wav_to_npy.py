import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import gc

def process_chunk(chunk, orig_sr, target_sr=44100, target_channels=16, max_duration=3.0):
    """Process individual audio chunk with memory efficiency"""
    try:
        # Resample if needed
        if orig_sr != target_sr:
            chunk = librosa.resample(chunk.T, orig_sr=orig_sr, target_sr=target_sr).T
        
        # Convert to target channels
        chunk = convert_to_16ch(chunk)
        
        # Trim/pad to exact duration
        target_samples = int(target_sr * max_duration)
        if chunk.shape[0] > target_samples:
            return chunk[:target_samples]
        return np.pad(chunk, ((0, target_samples - chunk.shape[0]), (0, 0)))
    
    except Exception as e:
        logging.error(f"Chunk processing failed: {str(e)}")
        return None

def convert_to_16ch(wav_array):
    """Channel conversion logic with minimal memory footprint"""
    if wav_array.ndim == 1:
        wav_array = wav_array[:, np.newaxis]
        
    num_channels = wav_array.shape[1]
    
    if num_channels == 1:
        return np.repeat(wav_array, 16, axis=1)
    elif 2 <= num_channels < 16:
        repeats = 16 // num_channels
        remainder = 16 % num_channels
        return np.hstack([wav_array] * repeats + [wav_array[:, :remainder]])
    else:
        return wav_array[:, :16]

def process_audio(input_path, output_dir, chunk_size_sec=10, max_duration=3.0):
    """Process audio file in memory-efficient chunks"""
    try:
        with sf.SoundFile(input_path) as f:
            orig_sr = f.samplerate
            chunk_size = int(orig_sr * chunk_size_sec)
            
            for i, chunk in enumerate(f.blocks(blocksize=chunk_size, dtype='float32')):
                # Process chunk
                processed = process_chunk(chunk, orig_sr)
                
                if processed is not None:
                    # Save chunk immediately to free memory
                    output_path = output_dir / f"{Path(input_path).stem}_chunk{i}.npy"
                    np.save(output_path, processed.astype('float32'))
                
                # Explicit memory cleanup
                del chunk, processed
                gc.collect()
                
    except Exception as e:
        logging.error(f"Failed {input_path}: {str(e)}")

def convert_dataset(src_root, dest_root):
    """Main conversion function with memory safeguards"""
    Path(dest_root).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename='conversion_errors.log', level=logging.ERROR)
    
    # Disable librosa caching to save memory
    librosa.set_cache_enabled(False)
    
    categories = {
        "yes_drone": "clean_drone_16ch",
        "unknown": "noise_16ch"
    }
    
    for src_category, dest_category in categories.items():
        input_dir = Path(src_root) / src_category
        output_dir = Path(dest_root) / dest_category
        output_dir.mkdir(exist_ok=True)
        
        print(f"Processing {src_category}...")
        files = list(input_dir.glob("*.wav"))
        
        # Process files one at a time
        for wav_file in tqdm(files, desc="Converting files"):
            process_audio(wav_file, output_dir)
            gc.collect()  # Force cleanup between files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Memory-efficient WAV to NPY converter')
    parser.add_argument('--src-root', type=str, required=True,
                      help='Root directory containing yes_drone/unknown folders')
    parser.add_argument('--dest-root', type=str, default='data',
                      help='Output root directory (default: data)')
    
    args = parser.parse_args()
    
    convert_dataset(
        src_root=args.src_root,
        dest_root=args.dest_root
    )
