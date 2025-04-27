# src/utils.py

import torch
import torch.nn.functional as F

# --- STFT and iSTFT functions ---

def stft(wav, n_fft=1024, hop_length=512, win_length=1024):
    """Compute Short-Time Fourier Transform."""
    return torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(wav.device),
        return_complex=True
    )

def istft(stft_matrix, n_fft=1024, hop_length=512, win_length=1024):
    """Compute Inverse Short-Time Fourier Transform."""
    return torch.istft(
        stft_matrix,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(stft_matrix.device),
        length=None  # You can pass length if needed
    )

# --- SI-SDR Loss ---

def si_sdr_loss(pred, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss.
    pred and target are waveforms: (batch, time)
    """
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Projection
    s_target = (torch.sum(pred * target, dim=-1, keepdim=True) * target) / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    e_noise = pred - s_target

    # SI-SDR
    si_sdr = 10 * torch.log10((s_target ** 2).mean(dim=-1) / (e_noise ** 2).mean(dim=-1) + eps)

    return -si_sdr.mean()  # Negative because we want to minimize the loss

# --- Optional: SDR / SIR / SAR Metrics (placeholder) ---

def compute_sdr_sir_sar(reference, estimation):
    """
    Placeholder for SDR/SIR/SAR computation using mir_eval or custom.
    reference and estimation are numpy arrays.
    """
    import mir_eval

    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference[None, :], estimation[None, :])
    return sdr[0], sir[0], sar[0]


