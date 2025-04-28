# src/data_loader.py

import os
import random
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DroneMixtureDataset(Dataset):
    def __init__(self, data_dir, snr_levels=[-5, 0, 5, 10], sample_rate=16000, mode='train', split=0.8):
        """
        Dataset for drone sound separation.
        
        Args:
            data_dir (str): Base directory with data folders
            snr_levels (list): SNR levels used for mixing
            sample_rate (int): Audio sample rate
            mode (str): 'train' or 'test'
            split (float): Train/test split ratio (0.0-1.0)
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.snr_levels = snr_levels
        self.mode = mode
        
        # Base data directories
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        # Get the parent directory to access subdirectories
        parent_dir = os.path.dirname(data_dir)
        
        # Check metadata file
        metadata_path = os.path.join(parent_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata with {len(self.metadata)} entries")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            self.metadata = None
        
        # Mixture directory
        mixture_dir = os.path.join(parent_dir, "mixtures")
        clean_dir = os.path.join(parent_dir, "clean_drone")
        
        if not os.path.exists(mixture_dir):
            raise FileNotFoundError(f"Mixtures directory not found: {mixture_dir}")
        if not os.path.exists(clean_dir):
            raise FileNotFoundError(f"Clean drone directory not found: {clean_dir}")
            
        # Preload file paths
        self.mixture_paths = []
        self.clean_paths = []

        # Find all mixture files
        mixture_files = [f for f in os.listdir(mixture_dir) if f.endswith('.wav')]
        
        # Create train/test split
        random.seed(42)  # For reproducibility
        random.shuffle(mixture_files)
        split_idx = int(len(mixture_files) * split)
        
        if mode == 'train':
            file_list = mixture_files[:split_idx]
        else:  # test
            file_list = mixture_files[split_idx:]
        
        # For each mixture file, find corresponding clean file
        for fname in file_list:
            # Extract original clean file name from mixture filename
            # Assuming format: cleanfile_mixed_with_noisefile_snrX.wav
            clean_file = fname.split('_mixed_with_')[0] + '.wav'
            
            mixture_path = os.path.join(mixture_dir, fname)
            clean_path = os.path.join(clean_dir, clean_file)
            
            # Check if clean file exists
            if not os.path.exists(clean_path):
                logger.warning(f"Clean file not found: {clean_path}")
                continue
                
            self.mixture_paths.append(mixture_path)
            self.clean_paths.append(clean_path)

        if not self.mixture_paths:
            raise ValueError(f"No valid audio pairs found in {mixture_dir}")
            
        logger.info(f"Loaded {len(self.mixture_paths)} audio pairs for {mode}")

    def __len__(self):
        return len(self.mixture_paths)

    def __getitem__(self, idx):
        try:
            # Load noisy mixture
            mixture, sr = torchaudio.load(self.mixture_paths[idx])
            # Load clean drone sound
            clean, sr = torchaudio.load(self.clean_paths[idx])
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                mixture = resampler(mixture)
                clean = resampler(clean)

            # Random crop if too long
            target_len = self.sample_rate * 3  # 3 seconds
            if mixture.size(1) > target_len:
                start = random.randint(0, mixture.size(1) - target_len)
                mixture = mixture[:, start:start+target_len]
                clean = clean[:, start:start+target_len]
            else:
                # Pad if too short
                pad_len = target_len - mixture.size(1)
                mixture = torch.nn.functional.pad(mixture, (0, pad_len))
                clean = torch.nn.functional.pad(clean, (0, pad_len))

            return mixture.squeeze(0), clean.squeeze(0)  # Remove channel dim for simplicity
            
        except Exception as e:
            logger.error(f"Error loading files: {self.mixture_paths[idx]} / {self.clean_paths[idx]}")
            logger.error(f"Error details: {str(e)}")
            # Return a default item or try the next one
            if idx < len(self) - 1:
                return self.__getitem__(idx + 1)
            else:
                # Create zeros as fallback
                dummy = torch.zeros(self.sample_rate * 3)
                return dummy, dummy
