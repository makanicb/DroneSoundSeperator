# src/multi_channel_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import logging
from pathlib import Path
from collections import OrderedDict
import time
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiChannelDroneDataset(Dataset):
    """Optimized Dataset for multi-channel drone audio processing with caching."""
    
    def __init__(
        self,
        mixtures_dir: str,
        clean_dir: str,
        noise_dir: str,
        dataset_overview_path: str,
        sample_rate: int = 44100,
        chunk_size_seconds: float = 3.0,
        mode: str = "train",
        split: float = 0.8,
        cache_size: int = 100,  # Cache size in items
    ):
        # Store parameters as attributes
        self.sample_rate = sample_rate
        self.chunk_size_seconds = chunk_size_seconds
        self.cache = OrderedDict()  # LRU cache
        self.cache_size = cache_size
        self.target_length = int(self.chunk_size_seconds * self.sample_rate)

        # Load dataset_overview.json
        with open(dataset_overview_path, "r") as f:
            self.dataset_overview = json.load(f)  # List of {"session_id", "clean", "noise"}

        # Store directories
        self.mixtures_dir = Path(mixtures_dir)
        self.clean_dir = Path(clean_dir)
        self.noise_dir = Path(noise_dir)

        # Filter valid sessions (those that exist in the filesystem)
        self.sessions = []
        for entry in self.dataset_overview:
            session_path = self.mixtures_dir / entry["session_id"]
            if session_path.exists():
                self.sessions.append(entry)
            else:
                logging.warning(f"Session {entry['session_id']} not found in {self.mixtures_dir}")

        # Split into train/test
        split_idx = int(len(self.sessions) * split)
        if mode == "train":
            self.sessions = self.sessions[:split_idx]
        else:
            self.sessions = self.sessions[split_idx:]

        self.indices = list(range(len(self.sessions)))
        logging.info(f"Initialized dataset with {len(self.sessions)} {mode} samples (cache size: {cache_size})")
        
    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            # Move to end to mark as recently used
            data = self.cache.pop(idx)
            self.cache[idx] = data
            return data

        session = self.sessions[idx]
    
        try:
            # 1. Load mixed audio (shape: [samples, channels])
            mix_path = self.mixtures_dir / session["session_id"] / "audio_chunks" / "chunk_0.npy"
            mixed_audio = np.load(mix_path)
    
            # 2. Load corresponding clean audio (shape: [samples, channels])
            clean_path = self.clean_dir / Path(session["clean"]).name  # Extract filename from path
            clean_audio = np.load(clean_path)
    
            # 3. Ensure shapes match (pad/truncate if needed)
            mixed_audio = self._ensure_length(mixed_audio, self.target_length)
            clean_audio = self._ensure_length(clean_audio, self.target_length)
    
            # 4. Convert to tensors and normalize
            mixed_tensor = torch.from_numpy(mixed_audio).float().permute(1, 0)  # [C, S]
            clean_tensor = torch.from_numpy(clean_audio).float().permute(1, 0)  # [C, S]
    
            result = (mixed_tensor, clean_tensor)
            
            # Add to cache if there's space
            if len(self.cache) < self.cache_size:
                self.cache[idx] = result
            else:
                # Remove oldest item if cache is full
                self.cache.popitem(last=False)
                self.cache[idx] = result
                
            return result

        except Exception as e:
            logger.error(f"Error loading session {session['session_id']}: {str(e)}")
            # Return silent dummy audio with correct shape
            dummy_audio = torch.zeros(16, self.target_length)
            return dummy_audio, dummy_audio
    
    def _ensure_length(self, audio: np.ndarray, target_samples: int) -> np.ndarray:
        """Pad or truncate audio to target length."""
        current_samples = audio.shape[0]
        if current_samples > target_samples:
            # Truncate randomly
            start = np.random.randint(0, current_samples - target_samples)
            return audio[start : start + target_samples, :]
        elif current_samples < target_samples:
            # Pad with zeros
            padding = target_samples - current_samples
            return np.pad(audio, ((0, padding), (0, 0)), mode="constant")
        else:
            return audio

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

def create_data_loader(dataset, batch_size, num_workers, shuffle=False):
    """
    Create optimized DataLoader with caching, prefetching, and pinned memory.
    
    Args:
        dataset: Initialized dataset
        batch_size: Batch size
        num_workers: Number of parallel workers
        shuffle: Whether to shuffle data
        
    Returns:
        Optimized DataLoader instance
    """
    # Automatically reduce workers if memory constrained
    if psutil.virtual_memory().percent > 85:
        reduced_workers = max(1, num_workers // 2)
        logger.warning(f"High memory usage ({psutil.virtual_memory().percent}%). "
                       f"Reducing workers from {num_workers} to {reduced_workers}")
        num_workers = reduced_workers
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
