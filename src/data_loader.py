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
    ):
        # Store parameters as attributes
        self.sample_rate = sample_rate
        self.chunk_size_seconds = chunk_size_seconds

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
        logging.info(f"Initialized dataset with {len(self.sessions)} {mode} samples")
        
    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]
    
        try:
            # 1. Load mixed audio (shape: [samples, channels])
            mix_path = self.mixtures_dir / session["session_id"] / "audio_chunks" / "chunk_0.npy"
            mixed_audio = np.load(mix_path)
    
            # 2. Load corresponding clean audio (shape: [samples, channels])
            clean_path = self.clean_dir / Path(session["clean"]).name  # Extract filename from path
            clean_audio = np.load(clean_path)
    
            # 3. Ensure shapes match (pad/truncate if needed)
            target_length = int(self.chunk_size_seconds * self.sample_rate)
            mixed_audio = self._ensure_length(mixed_audio, target_length)
            clean_audio = self._ensure_length(clean_audio, target_length)
    
            # 4. Convert to tensors and normalize
            mixed_tensor = torch.from_numpy(mixed_audio).float().permute(1, 0)  # [C, S]
            clean_tensor = torch.from_numpy(clean_audio).float().permute(1, 0)  # [C, S]
    
            return mixed_tensor, clean_tensor

        except Exception as e:
            logger.error(f"Error loading session {session['session_id']}: {str(e)}")
            # Return silent dummy audio
            return torch.zeros(16, int(self.sample_rate * self.chunk_size_seconds)), metadata
    
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
