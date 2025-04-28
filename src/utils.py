# src/utils.py

import torch
import torch.nn.functional as F
import numpy as np

# --- STFT and iSTFT functions ---

def stft(wav, n_fft=1024, hop_length=512, win_length=1024):
    """
    Compute Short-Time Fourier Transform.
    
    Args:
        wav: Input waveform tensor of shape (batch_size, time)
        n_fft: FFT size
        hop_length: Hop size between frames
        win_length: Window length
        
    Returns:
        Complex STFT tensor of shape (batch_size, freq_bins, time_frames)
    """
    # Ensure waveform has batch dimension
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    window = torch.hann_window(win_length).to(wav.device)
    
    # Compute STFT
    stft_matrix = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    
    return stft_matrix


def istft(stft_matrix, n_fft=1024, hop_length=512, win_length=1024, length=None):
    """
    Compute Inverse Short-Time Fourier Transform.
    
    Args:
        stft_matrix: Complex STFT tensor of shape (batch_size, freq_bins, time_frames)
        n_fft: FFT size
        hop_length: Hop size between frames
        win_length: Window length
        length: Optional target length for the output waveform
        
    Returns:
        Reconstructed waveform tensor of shape (batch_size, time)
    """
    window = torch.hann_window(win_length).to(stft_matrix.device)
    
    # Compute inverse STFT
    wav = torch.istft(
        stft_matrix,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length
    )
    
    return wav


# --- SI-SDR Loss ---

def si_sdr_loss(pred, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss.
    
    Args:
        pred: Predicted waveform of shape (batch_size, time)
        target: Target waveform of shape (batch_size, time)
        eps: Small constant for numerical stability
        
    Returns:
        Negative SI-SDR loss (to minimize)
    """
    # Remove mean (DC component)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Projection
    s_target = (torch.sum(pred * target, dim=-1, keepdim=True) * target) / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    e_noise = pred - s_target

    # SI-SDR
    si_sdr = 10 * torch.log10((s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps)

    return -si_sdr.mean()  # Negative because we want to minimize the loss


# --- SDR / SIR / SAR Metrics ---

def compute_sdr_sir_sar(reference, estimation, compute_permutation=False):
    """
    Compute Source-to-Distortion Ratio (SDR), Source-to-Interference Ratio (SIR),
    and Source-to-Artifacts Ratio (SAR) using mir_eval.
    
    Args:
        reference: Reference source signal (numpy array)
        estimation: Estimated source signal (numpy array)
        compute_permutation: Whether to compute the best permutation
        
    Returns:
        SDR, SIR, SAR values in dB
    """
    try:
        import mir_eval.separation
        
        # Ensure signals are 2D (sources x samples)
        if reference.ndim == 1:
            reference = reference[np.newaxis, :]
        if estimation.ndim == 1:
            estimation = estimation[np.newaxis, :]
        
        # Compute metrics
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            reference,
            estimation,
            compute_permutation=compute_permutation
        )
        
        return sdr[0], sir[0], sar[0]
    
    except ImportError:
        print("mir_eval package not found. Please install it with: pip install mir_eval")
        return 0.0, 0.0, 0.0
    except Exception as e:
        print(f"Error computing separation metrics: {e}")
        return 0.0, 0.0, 0.0


# --- Visualization Helper Functions ---

def plot_waveform(waveform, sample_rate=16000, title="Waveform"):
    """
    Plot a waveform using matplotlib.
    
    Args:
        waveform: Audio waveform (numpy array or tensor)
        sample_rate: Audio sample rate
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    if torch.is_tensor(waveform):
        waveform = waveform.cpu().numpy()
    
    plt.figure(figsize=(10, 3))
    plt.plot(waveform)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    return plt.gcf()


def plot_spectrogram(spectrogram, title="Spectrogram"):
    """
    Plot a spectrogram using matplotlib.
    
    Args:
        spectrogram: Spectrogram (numpy array or tensor)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import librosa.display
    
    if torch.is_tensor(spectrogram):
        spectrogram = spectrogram.cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(spectrogram, ref=np.max),
        y_axis='log',
        x_axis='time'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()
