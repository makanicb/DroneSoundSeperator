# src/data_loader.py

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

class DroneMixtureDataset(Dataset):
    def __init__(self, data_dir, snr_levels=[-5, 0, 5, 10], sample_rate=16000):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.snr_levels = snr_levels

        # Preload file paths
        self.mixture_paths = []
        self.clean_paths = []

        for fname in os.listdir(os.path.join(data_dir, "mixtures")):
            if fname.endswith(".wav"):
                self.mixture_paths.append(os.path.join(data_dir, "mixtures", fname))
                self.clean_paths.append(os.path.join(data_dir, "clean_drone", fname))

        assert len(self.mixture_paths) == len(self.clean_paths), "Mismatch between mixtures and clean files!"

    def __len__(self):
        return len(self.mixture_paths)

    def __getitem__(self, idx):
        # Load noisy mixture
        mixture, _ = torchaudio.load(self.mixture_paths[idx])
        # Load clean drone sound
        clean, _ = torchaudio.load(self.clean_paths[idx])

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


