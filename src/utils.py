# src/multi_channel_utils.py

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os

def stft(wav, n_fft=2048, hop_length=441, win_length=2048):
    """
    Compute Short-Time Fourier Transform for multi-channel audio.
    
    Args:
        wav: Input waveform tensor of shape (batch_size, channels, time)
        n_fft: FFT size
        hop_length: Hop size between frames
        win_length: Window length
        
    Returns:
        Complex STFT tensor of shape (batch_size, channels, freq_bins, time_frames)
    """
    batch_size, num_channels, time_length = wav.shape
    window = torch.hann_window(win_length).to(wav.device)
    
    # Create output tensor to store results
    stft_results = []
    
    # Process each channel separately
    for ch in range(num_channels):
        stft_ch = torch.stft(
            wav[:, ch, :],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        stft_results.append(stft_ch)
    
    # Stack channels
    stft_tensor = torch.stack(stft_results, dim=1)
    
    return stft_tensor

def istft(stft_matrix, n_fft=2048, hop_length=441, win_length=2048, length=None):
    """
    Compute Inverse Short-Time Fourier Transform for multi-channel audio.
    
    Args:
        stft_matrix: Complex STFT tensor of shape (batch_size, channels, freq_bins, time_frames)
        n_fft: FFT size
        hop_length: Hop size between frames
        win_length: Window length
        length: Optional target length for the output waveform
        
    Returns:
        Reconstructed waveform tensor of shape (batch_size, channels, time)
    """
    batch_size, num_channels = stft_matrix.shape[0], stft_matrix.shape[1]
    window = torch.hann_window(win_length).to(stft_matrix.device)
    
    # Process each channel separately
    wav_results = []
    
    for ch in range(num_channels):
        wav_ch = torch.istft(
            stft_matrix[:, ch],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=length
        )
        wav_results.append(wav_ch)
    
    # Stack channels
    wav_tensor = torch.stack(wav_results, dim=1)
    
    return wav_tensor

def si_sdr_loss(pred, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss for multi-channel audio.
    
    Args:
        pred: Predicted waveform of shape (batch_size, channels, time)
        target: Target waveform of shape (batch_size, channels, time)
        eps: Small constant for numerical stability
        
    Returns:
        Negative SI-SDR loss (to minimize)
    """
    # Remove mean (DC component) per channel
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Compute SI-SDR for each channel
    batch_size, num_channels = pred.shape[0], pred.shape[1]
    si_sdr_values = []
    
    for b in range(batch_size):
        for ch in range(num_channels):
            pred_ch = pred[b, ch]
            target_ch = target[b, ch]
            
            # Projection
            s_target = ((pred_ch * target_ch).sum() * target_ch) / ((target_ch ** 2).sum() + eps)
            e_noise = pred_ch - s_target
            
            # SI-SDR
            si_sdr_ch = 10 * torch.log10((s_target ** 2).sum() / ((e_noise ** 2).sum() + eps) + eps)
            si_sdr_values.append(si_sdr_ch)
    
    # Average over all channels and batches
    si_sdr_loss = -torch.stack(si_sdr_values).mean()
    
    return si_sdr_loss

def compute_sdr_sir_sar(reference, estimation):
    """
    Compute metrics for multi-channel source separation.
    
    Args:
        reference: Reference source signals [channels, samples]
        estimation: Estimated source signals [channels, samples]
        
    Returns:
        Dictionary of metrics
    """
    if torch.is_tensor(reference):
        reference = reference.detach().cpu().numpy()
    if torch.is_tensor(estimation):
        estimation = estimation.detach().cpu().numpy()
    
    # Make sure they are numpy arrays
    reference = np.asarray(reference)
    estimation = np.asarray(estimation)
    
    # Initialize metrics
    sdr_values = []
    sir_values = []
    sar_values = []
    
    num_channels = reference.shape[0]

    try:
        import mir_eval.separation
        
        # For each channel, compute metrics
        for ch in range(num_channels):
            ref_ch = reference[ch:ch+1]  # Keep dimension for mir_eval
            est_ch = estimation[ch:ch+1]  # Keep dimension for mir_eval
            
            # Compute metrics
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                ref_ch,
                est_ch,
                compute_permutation=False
            )
            
            sdr_values.append(sdr[0])
            sir_values.append(sir[0])
            sar_values.append(sar[0])
        
        # Average metrics
        avg_sdr = np.mean(sdr_values)
        avg_sir = np.mean(sir_values)
        avg_sar = np.mean(sar_values)
        
        return {
            'sdr': avg_sdr,
            'sir': avg_sir,
            'sar': avg_sar,
            'sdr_per_channel': sdr_values,
            'sir_per_channel': sir_values,
            'sar_per_channel': sar_values
        }
        
    except ImportError:
        print("mir_eval package not found. Please install it with: pip install mir_eval")
        return {
            'sdr': 0.0,
            'sir': 0.0,
            'sar': 0.0
        }
    except Exception as e:
        print(f"Error computing separation metrics: {e}")
        return {
            'sdr': 0.0,
            'sir': 0.0,
            'sar': 0.0
        }

def save_audio(audio_tensor, output_path, sample_rate=16000):
    """
    Save multi-channel audio tensor to a file.
    
    Args:
        audio_tensor: Audio tensor of shape [channels, samples]
        output_path: Path to save audio file
        sample_rate: Sample rate
    """
    # Convert to numpy
    if torch.is_tensor(audio_tensor):
        audio_tensor = audio_tensor.detach().cpu().numpy()
    
    # Check if output is .npy
    if output_path.endswith('.npy'):
        # Save as numpy array
        np.save(output_path, audio_tensor)
        print(f"Multi-channel audio saved to {output_path}")
    else:
        # For other formats, use soundfile/librosa
        try:
            
            # Save as multi-channel wav
            # Transpose to [samples, channels] for soundfile
            audio_array = audio_tensor.T
            sf.write(output_path, audio_array, sample_rate)
            print(f"Multi-channel audio saved to {output_path}")
        except Exception as e:
            print(f"Error saving audio: {e}")
            # Fallback to numpy save
            np_path = output_path.rsplit('.', 1)[0] + '.npy'
            np.save(np_path, audio_tensor)
            print(f"Failed to save in requested format. Saved as numpy array to {np_path}")

def save_comparison_samples(clean, mixed, estimate, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (c, m, e) in enumerate(zip(clean, mixed, estimate)):
        print(f"Clean shape: {c.shape}")
        sf.write(f"{output_dir}/sample_{i}_clean.wav", c.cpu().numpy().T, 44100)
        sf.write(f"{output_dir}/sample_{i}_mixed.wav", m.cpu().numpy().T, 44100)
        sf.write(f"{output_dir}/sample_{i}_estimate.wav", e.cpu().numpy().T, 44100)
