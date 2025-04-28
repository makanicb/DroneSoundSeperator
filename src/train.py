# src/train.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime

from data_loader import DroneMixtureDataset
from model import UNetSeparator
from utils import stft, istft, si_sdr_loss, compute_sdr_sir_sar

def train(config_path):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    os.makedirs(config['checkpoints']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=config['logging']['log_dir']) if config['logging']['use_tensorboard'] else None
    
    # Setup datasets
    train_dataset = DroneMixtureDataset(
        data_dir=config['data_dir'],
        snr_levels=config['snr_levels'],
        sample_rate=config['sample_rate'],
        mode='train',
        split=config['train_test_split']
    )
    
    val_dataset = DroneMixtureDataset(
        data_dir=config['data_dir'],
        snr_levels=config['snr_levels'],
        sample_rate=config['sample_rate'],
        mode='test',
        split=config['train_test_split']
    )
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = UNetSeparator(
        in_channels=config['model']['in_channels'],
        base_channels=config['model']['base_channels']
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['lr'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    
    # Setup learning rate scheduler
    scheduler = None
    if config['training']['lr_scheduler']['use']:
        if config['training']['lr_scheduler']['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['training']['lr_scheduler']['step_size'],
                gamma=config['training']['lr_scheduler']['gamma']
            )
        elif config['training']['lr_scheduler']['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs']
            )
        elif config['training']['lr_scheduler']['type'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config['training']['lr_scheduler']['gamma'],
                patience=config['training']['lr_scheduler']['patience'],
                verbose=True
            )
    
    # Early stopping setup
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = config['training']['early_stopping']['patience']
    early_stopping_delta = config['training']['early_stopping']['min_delta']
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (noisy_wav, clean_wav) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")):
            noisy_wav = noisy_wav.to(device)
            clean_wav = clean_wav.to(device)
            
            # STFT
            X = stft(
                noisy_wav,
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length']
            )
            
            # Model expects magnitude spectrograms with channel dimension
            X_mag = torch.abs(X).unsqueeze(1)
            
            # Predict mask
            mask = model(X_mag)
            
            # Apply mask and reconstruct
            est_mag = mask * X_mag.squeeze(1)
            est_spec = est_mag * torch.exp(1j * torch.angle(X))
            est_wav = istft(
                est_spec,
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length']
            )
            
            # Calculate loss
            loss = si_sdr_loss(est_wav, clean_wav)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            if writer and global_step % config['logging']['image_freq'] == 0:
                # Log spectrograms
                writer.add_figure(
                    'spectrograms/train',
                    plot_spectrograms(noisy_wav[0].cpu(), clean_wav[0].cpu(), est_wav[0].cpu()),
                    global_step
                )
                writer.add_scalar('loss/train_step', loss.item(), global_step)
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        if epoch % config['validation']['interval'] == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_metrics = {'sdr': [], 'sir': [], 'sar': []}
            
            with torch.no_grad():
                for noisy_wav, clean_wav in tqdm(val_loader, desc="Validation"):
                    noisy_wav = noisy_wav.to(device)
                    clean_wav = clean_wav.to(device)
                    
                    # STFT
                    X = stft(
                        noisy_wav,
                        n_fft=config['n_fft'],
                        hop_length=config['hop_length'],
                        win_length=config['win_length']
                    )
                    
                    # Model expects magnitude spectrograms with channel dimension
                    X_mag = torch.abs(X).unsqueeze(1)
                    
                    # Predict mask
                    mask = model(X_mag)
                    
                    # Apply mask and reconstruct
                    est_mag = mask * X_mag.squeeze(1)
                    est_spec = est_mag * torch.exp(1j * torch.angle(X))
                    est_wav = istft(
                        est_spec,
                        n_fft=config['n_fft'],
                        hop_length=config['hop_length'],
                        win_length=config['win_length']
                    )
                    
                    # Calculate loss
                    loss = si_sdr_loss(est_wav, clean_wav)
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Calculate separation metrics for a subset
                    if val_batches % 10 == 0:  # Only compute on every 10th batch to save time
                        for i in range(min(2, clean_wav.size(0))):  # Only use first 2 samples from batch
                            clean_np = clean_wav[i].cpu().numpy()
                            est_np = est_wav[i].cpu().numpy()
                            
                            # Compute metrics
                            try:
                                sdr, sir, sar = compute_sdr_sir_sar(clean_np, est_np)
                                val_metrics['sdr'].append(sdr)
                                val_metrics['sir'].append(sir)
                                val_metrics['sar'].append(sar)
                            except Exception as e:
                                print(f"Error computing metrics: {e}")
            
            # Calculate average validation loss and metrics
            avg_val_loss = val_loss / val_batches
            avg_sdr = np.mean(val_metrics['sdr']) if val_metrics['sdr'] else 0
            avg_sir = np.mean(val_metrics['sir']) if val_metrics['sir'] else 0
            avg_sar = np.mean(val_metrics['sar']) if val_metrics['sar'] else 0
            
            # Print results
            print(f"Epoch {epoch+1}/{config['training']['epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"SDR: {avg_sdr:.2f} dB | "
                  f"SIR: {avg_sir:.2f} dB | "
                  f"SAR: {avg_sar:.2f} dB")
            
            # Log to tensorboard
            if writer:
                writer.add_scalar('loss/train', avg_train_loss, epoch)
                writer.add_scalar('loss/val', avg_val_loss, epoch)
                writer.add_scalar('metrics/sdr', avg_sdr, epoch)
                writer.add_scalar('metrics/sir', avg_sir, epoch)
                writer.add_scalar('metrics/sar', avg_sar, epoch)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                
                # Log validation spectrograms
                if val_loader:
                    noisy_sample, clean_sample = next(iter(val_loader))
                    noisy_sample = noisy_sample[:1].to(device)  # Just use first sample
                    clean_sample = clean_sample[:1].to(device)
                    
                    # Process through model
                    X = stft(noisy_sample, n_fft=config['n_fft'], hop_length=config['hop_length'], win_length=config['win_length'])
                    X_mag = torch.abs(X).unsqueeze(1)
                    mask = model(X_mag)
                    est_mag = mask * X_mag.squeeze(1)
                    est_spec = est_mag * torch.exp(1j * torch.angle(X))
                    est_sample = istft(est_spec, n_fft=config['n_fft'], hop_length=config['hop_length'], win_length=config['win_length'])
                    
                    writer.add_figure(
                        'spectrograms/val',
                        plot_spectrograms(noisy_sample[0].cpu(), clean_sample[0].cpu(), est_sample[0].cpu()),
                        epoch
                    )
            
            # Update learning rate scheduler
            if scheduler:
                if config['training']['lr_scheduler']['type'] == 'plateau':
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                config['checkpoints']['save_dir'],
                f"ckpt_epoch{epoch+1}.pt"
            )
            
            # Save model
            if config['checkpoints']['save_best_only']:
                if avg_val_loss < best_val_loss - early_stopping_delta:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                        'metrics': {
                            'sdr': avg_sdr,
                            'sir': avg_sir,
                            'sar': avg_sar
                        }
                    }, checkpoint_path)
                else:
                    early_stopping_counter += 1
                    print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            else:
                # Save model periodically
                if (epoch + 1) % config['checkpoints']['save_frequency'] == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                        'metrics': {
                            'sdr': avg_sdr,
                            'sir': avg_sir,
                            'sar': avg_sar
                        }
                    }, checkpoint_path)
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    # Training finished
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")
    
    # Save final model
    final_path = os.path.join(config['checkpoints']['save_dir'], "final_model.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss if 'avg_val_loss' in locals() else None,
        'metrics': {
            'sdr': avg_sdr if 'avg_sdr' in locals() else None,
            'sir': avg_sir if 'avg_sir' in locals() else None,
            'sar': avg_sar if 'avg_sar' in locals() else None
        }
    }, final_path)
    
    print(f"Final model saved to {final_path}")
    
    # Close tensorboard writer
    if writer:
        writer.close()

def plot_spectrograms(noisy, clean, estimated):
    """Create a figure with spectrograms for visualization in tensorboard."""
    import matplotlib.pyplot as plt
    from librosa.display import specshow
    import librosa
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    
    # Convert to numpy
    noisy = noisy.numpy()
    clean = clean.numpy()
    estimated = estimated.numpy()
    
    # Compute spectrograms
    noisy_spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(noisy)), ref=np.max
    )
    clean_spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(clean)), ref=np.max
    )
    est_spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(estimated)), ref=np.max
    )
    
    # Plot spectrograms
    specshow(noisy_spec, y_axis='log', x_axis='time', ax=axes[0])
    axes[0].set_title('Noisy Mixture')
    
    specshow(clean_spec, y_axis='log', x_axis='time', ax=axes[1])
    axes[1].set_title('Clean Reference')
    
    specshow(est_spec, y_axis='log', x_axis='time', ax=axes[2])
    axes[2].set_title('Estimated Clean')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    train(args.config)
