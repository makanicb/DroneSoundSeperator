# src/multi_channel_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiChannelDroneDataset(Dataset):
    """Dataset for multi-channel drone audio processing."""
    
    def __init__(self, data_dir, sample_rate=44100, chunk_size_seconds=3.0, mode='train', split=0.8):
        """
        Initialize the multi-channel drone dataset.
        
        Args:
            data_dir (str): Directory containing session folders with audio_chunks and metadata
            sample_rate (int): Audio sample rate
            chunk_size_seconds (float): Length of audio chunks in seconds
            mode (str): 'train' or 'test'
            split (float): Train/test split ratio (0.0-1.0)
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.chunk_size_seconds = chunk_size_seconds
        self.chunk_size_samples = int(chunk_size_seconds * sample_rate)
        self.mode = mode
        
        # Find all session directories
        self.session_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("20")]
        logger.info(f"Found {len(self.session_dirs)} session directories")
        
        # Load metadata and audio chunks from each session
        self.audio_chunks = []
        self.metadata_list = []
        
        for session_dir in self.session_dirs:
            metadata_file = session_dir / "metadata.json"
            if not metadata_file.exists():
                logger.warning(f"No metadata.json found in {session_dir}")
                continue
                
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Get audio chunks
                chunks_dir = session_dir / "audio_chunks"
                if not chunks_dir.exists():
                    logger.warning(f"No audio_chunks directory found in {session_dir}")
                    continue
                    
                chunk_files = sorted([f for f in chunks_dir.glob("*.npy")])
                
                # Map each chunk to its metadata information
                for chunk_file in chunk_files:
                    chunk_index = int(chunk_file.stem.split('_')[-1])
                    
                    # Find corresponding chunk metadata
                    chunk_meta = None
                    for chunk in metadata.get('audio_chunks_timestamps', []):
                        if chunk.get('chunk_index') == chunk_index:
                            chunk_meta = chunk
                            break
                    
                    if chunk_meta:
                        self.audio_chunks.append(chunk_file)
                        
                        # Copy session metadata and add chunk-specific metadata
                        combined_meta = metadata.copy()
                        combined_meta['chunk_metadata'] = chunk_meta
                        self.metadata_list.append(combined_meta)
                
            except Exception as e:
                logger.error(f"Error processing session {session_dir}: {str(e)}")
        
        logger.info(f"Loaded {len(self.audio_chunks)} audio chunks across all sessions")
        
        # Split data into train/test
        indices = list(range(len(self.audio_chunks)))
        split_idx = int(len(indices) * split)
        
        if mode == 'train':
            self.indices = indices[:split_idx]
        else:  # test
            self.indices = indices[split_idx:]
            
        logger.info(f"Using {len(self.indices)} chunks for {mode}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get an audio chunk.
        
        Returns:
            multi_channel_audio: Tensor of shape [channels, samples]
            metadata: Dictionary with metadata about the chunk
        """
        index = self.indices[idx]
        chunk_path = self.audio_chunks[index]
        metadata = self.metadata_list[index]
    
        try:
            audio = np.load(chunk_path)
        
            # Enforce (samples, channels) format
            if audio.shape[0] < audio.shape[1]:  # Assume (C,S) if C < S
                audio = audio.T
        
            # Validate shape
            assert audio.ndim == 2, f"Audio must be 2D (got shape {audio.shape})"
            assert audio.shape[1] in [1, 16], f"Expected 1/16 channels (got {audio.shape[1]})"
        
            # Convert to tensor and normalize
            audio_tensor, _ = self.process_audio(audio)
        
            # Handle length
            target_samples = int(self.sample_rate * self.chunk_size_seconds)
            if audio_tensor.shape[0] > target_samples:
                start = torch.randint(0, audio_tensor.shape[0] - target_samples, (1,))
                audio_tensor = audio_tensor[start:start+target_samples, :]
            elif audio_tensor.shape[0] < target_samples:
                padding = target_samples - audio_tensor.shape[0]
                audio_tensor = F.pad(audio_tensor, (0, 0, 0, padding))
            
            return audio_tensor.permute(1, 0), metadata  # (C,S) format expected by model

        except Exception as e:
            logger.error(f"Error loading {chunk_path}: {str(e)}")
            # Return silent dummy audio
            return torch.zeros(16, int(self.sample_rate * self.chunk_size_seconds)), metadata

    def process_audio(self, audio):
        """
        Process audio data: handle clipping, normalize, and prepare for model input.
    
        Args:
            audio: numpy array of audio data with shape (samples, channels)
    
        Returns:
            processed_audio: Tensor of shape (samples, channels) normalized to [-1.0, 1.0]
            stats: Dictionary containing processing statistics (optional)
        """
        # Ensure numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
    
        # Handle clipping
        max_value = np.max(np.abs(audio))
        was_clipped = max_value > 1.0
    
        # Convert to tensor and normalize
        audio_tensor = torch.from_numpy(audio).float()
        audio_tensor = audio_tensor / (torch.max(torch.abs(audio_tensor)) + 1e-8)
    
        # Optional: track processing stats
        stats = {
            'was_clipped': was_clipped,
            'original_max': max_value,
            'processed_max': torch.max(torch.abs(audio_tensor)).item()
        }
    
        return audio_tensor, stats

def load_and_preprocess_npy(file_path, target_sr=44100):
    """
    Load a .npy file containing multi-channel audio data.
    
    Args:
        file_path: Path to the .npy file
        target_sr: Target sample rate
        
    Returns:
        Tensor of shape [channels, samples]
    """
    try:
        # Load numpy array
        multi_channel_audio = np.load(file_path)
        
        # Transpose if needed
        if multi_channel_audio.shape[0] > multi_channel_audio.shape[1]:
            multi_channel_audio = multi_channel_audio.T
            
        # Normalize if needed
        if multi_channel_audio.dtype != np.float32:
            if multi_channel_audio.dtype == np.int16:
                multi_channel_audio = multi_channel_audio.astype(np.float32) / 32768.0
            elif multi_channel_audio.dtype == np.int32:
                multi_channel_audio = multi_channel_audio.astype(np.float32) / 2147483648.0
            elif multi_channel_audio.dtype == np.uint8:
                multi_channel_audio = (multi_channel_audio.astype(np.float32) - 128) / 128.0
        
        # Ensure it's normalized
        max_val = np.max(np.abs(multi_channel_audio))
        if max_val > 1.0:
            multi_channel_audio = multi_channel_audio / max_val
            
        # Convert to tensor
        audio_tensor = torch.tensor(multi_channel_audio, dtype=torch.float32)
        
        return audio_tensor
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None
