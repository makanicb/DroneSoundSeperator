# src/train.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from datetime import datetime

from src.data_loader import MultiChannelDroneDataset
from src.model import UNetSeparator
from src.utils import stft, istft, si_sdr_loss, compute_sdr_sir_sar

def train(config_path, resume_checkpoint=None, max_steps=None, max_val_steps=None):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config['checkpoints']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Initialize model
    model = UNetSeparator(
        input_channels=config['model']['in_channels'],
        base_channels=config['model']['base_channels']
    ).to(device)
    
    # Setup optimizer
    optimizer = create_optimizer(config, model)
    # Initialize scheduler only if enabled
    if config["training"]["lr_scheduler"]["use"]:
        scheduler = create_scheduler(config, optimizer)
    else:
        scheduler = None
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler(enabled=config['training']['use_amp'])
    
    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    # Resume training
    if resume_checkpoint:
        load_checkpoint(resume_checkpoint, device, model, optimizer, scheduler, 
                       config, start_epoch, best_val_loss, early_stopping_counter)

    # Data loaders
    train_loader, val_loader = create_data_loaders(config, device)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=config['logging']['log_dir']) if config['logging']['use_tensorboard'] else None
    
    # Training loop
    total_steps = 0
    for epoch in range(start_epoch, config['training']['epochs']):
        # Training phase
        model.train()
        train_loss, steps_completed, optimizer_steps = train_epoch(model, train_loader, optimizer, scaler, device, 
                                config, gradient_accumulation_steps, writer, epoch,
                                max_steps, total_steps)

        # Validation phase
        if epoch % config['validation']['interval'] == 0:
            val_loss, metrics = validate(model, val_loader, device, config, writer, epoch, max_val_steps)
            
            # Checkpoint handling
            best_val_loss, early_stopping_counter = handle_checkpoints(
                config, model, optimizer, scheduler, epoch, val_loss, 
                best_val_loss, early_stopping_counter, metrics
            )
            
            # Early stopping
            if early_stopping_counter >= config['training']['early_stopping']['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Learning rate scheduling
        if scheduler and optimizer_steps > 0: # Only update if optimizer stepped
            update_scheduler(scheduler, config, val_loss)

        # Exit if max_steps reached
        total_steps += steps_completed 
        
        if max_steps is not None and total_steps >= max_steps:
            print(f"Stopping early at {total_steps} steps")
            break
        
    # Final cleanup
    save_final_model(config, model, optimizer, epoch)
    if writer: writer.close()

def create_data_loaders(config, device):
    train_dataset = MultiChannelDroneDataset(
        mixtures_dir=config["data"]["mixtures_dir"],
        clean_dir=config["data"]["clean_dir"],
        noise_dir=config["data"]["noise_dir"],
        dataset_overview_path=config["data"]["dataset_overview"],
        sample_rate=config["data"]["sample_rate"],
        chunk_size_seconds=config["data"]["max_audio_length"],
        mode="train",
        split=config["data"]["train_test_split"],
    )

    val_dataset = MultiChannelDroneDataset(
        mixtures_dir=config["data"]["mixtures_dir"],
        clean_dir=config["data"]["clean_dir"],
        noise_dir=config["data"]["noise_dir"],
        dataset_overview_path=config["data"]["dataset_overview"],
        sample_rate=config["data"]["sample_rate"],
        chunk_size_seconds=config["data"]["max_audio_length"],
        mode="test",
        split=config["data"]["train_test_split"],
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["validation"]["batch_size"])
    return train_loader, val_loader

def create_optimizer(config, model):
    optimizer_type = config["training"]["optimizer"]  # Path: training → optimizer
    lr = config["training"]["lr"]                    # From training.lr
    weight_decay = config["training"]["weight_decay"]

    params = model.parameters()

    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,       # Hardcoded (add to config if needed)
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def create_scheduler(config, optimizer):
    # Corrected path to match config.yaml structure
    scheduler_config = config["training"]["lr_scheduler"]
    scheduler_type = scheduler_config["type"]
    
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5)
        )
    elif scheduler_type == "cosine":  # Match config's "cosine" type (not "CosineAnnealing")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("step_size", 10)
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("gamma", 0.5)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

def update_scheduler(scheduler, config, val_loss=None):
    if scheduler is None:
        return
    
    # For plateau schedulers, step with validation metric
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()

def train_epoch(model, loader, optimizer, scaler, device, config, grad_accum, writer, epoch, max_steps=None, total_steps=0):
    train_loss = 0.0
    optimizer.zero_grad()
    steps_completed = 0
    optimizer_steps = 0
    
    for batch_idx, (mixed, clean) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
        # Exit early if max_steps reached
        if max_steps is not None and (total_steps + steps_completed) >= max_steps:
            break

        mixed = mixed.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=config['training']['use_amp']):
            est_waveform = model(mixed)
            loss = si_sdr_loss(est_waveform, clean) / grad_accum
            
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if config['device'] == "cpu":
                free_memory()
            optimizer_steps += 1

        train_loss += loss.item()
        log_training(writer, epoch*len(loader)+batch_idx, loss.item())
 
        steps_completed += 1
        
    return train_loss / len(loader), steps_completed, optimizer_steps

def validate(model, loader, device, config, writer, epoch, max_val_steps=None):
    model.eval()
    val_loss = 0.0
    metrics = {'sdr': [], 'sir': [], 'sar': []}

    with torch.no_grad():
        for batch_idx, (mixed, clean) in enumerate(loader):

            # Exit early if max_val_steps is set
            if max_val_steps is not None and batch_idx >= max_val_steps:
                break            

            mixed = mixed.to(device)
            clean = clean.to(device)

            with torch.cuda.amp.autocast(enabled=config['validation']['use_amp']):
                est_waveform = model(mixed)
                loss = si_sdr_loss(est_waveform, clean)
                val_loss += loss.item()
                update_metrics(metrics, clean, est_waveform)  # Compute SDR/SIR/SAR

    log_validation(writer, epoch, val_loss / len(loader), metrics)
    return val_loss / len(loader), metrics

def handle_checkpoints(config, model, optimizer, scheduler, epoch, val_loss, 
                      best_val_loss, early_stopping_counter, metrics):
    if val_loss < best_val_loss - config['training']['early_stopping']['min_delta']:
        best_val_loss = val_loss
        early_stopping_counter = 0
        save_checkpoint(config, model, optimizer, scheduler, epoch, 
                       best_val_loss, early_stopping_counter, metrics, "best")
    else:
        early_stopping_counter += 1
        
    if (epoch + 1) % config['checkpoints']['save_frequency'] == 0:
        save_checkpoint(config, model, optimizer, scheduler, epoch, 
                       best_val_loss, early_stopping_counter, metrics, "latest")
        
    return best_val_loss, early_stopping_counter

def save_checkpoint(config, model, optimizer, scheduler, epoch, 
                   best_loss, early_stop_count, metrics, suffix):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_loss,
        'metrics': metrics,
        'early_stopping_counter': early_stop_count,
        'rng_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate()
        },
        'config': config
    }
    
    path = os.path.join(config['checkpoints']['save_dir'], f"ckpt_{suffix}.pt")
    torch.save(checkpoint, path)
    print(f"Saved {suffix} checkpoint to {path}")

def load_checkpoint(checkpoint_path, device, model, optimizer, scheduler, 
                   config, start_epoch, best_val_loss, early_stopping_counter):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    early_stopping_counter = checkpoint['early_stopping_counter']
    print(f"Resuming from epoch {start_epoch}")

# Mock implementations to avoid errors (customize later)
def log_training(writer, step, loss):
    if writer:
        writer.add_scalar("Loss/train", loss, step)

def log_validation(writer, epoch, val_loss, metrics):
    if writer:
        writer.add_scalar("Loss/val", val_loss, epoch)

def update_metrics(metrics_dict, clean, estimate):
    # Mock metric calculation
    metrics_dict['sdr'].append(0.0)
    metrics_dict['sir'].append(0.0)
    metrics_dict['sar'].append(0.0)

def free_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_final_model(config, model, optimizer, epoch):
    """Saves model at end of training (even if interrupted early)."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    path = os.path.join(config['checkpoints']['save_dir'], "final_model.pt")
    torch.save(checkpoint, path)
    print(f"Saved final model to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, help='Checkpoint path to resume from')
    parser.add_argument('--max_steps', type=int, default=None, help='Limit training to N steps')
    parser.add_argument('--max_val_steps', type=int, default=None, help='Limit validation to N batches')
    args = parser.parse_args()
    
    train(args.config, args.resume, args.max_steps, args.max_val_steps)
