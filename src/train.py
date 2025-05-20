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

from data_loader import MultiChannelDroneDataset
from model import UNetSeparator
from utils import stft, istft, si_sdr_loss, compute_sdr_sir_sar

def train(config_path, resume_checkpoint=None, max_steps=None):
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
    scheduler = create_scheduler(config, optimizer)
    
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
        train_loss, steps_completed = train_epoch(model, train_loader, optimizer, scaler, device, 
                                config, gradient_accumulation_steps, writer, epoch,
                                max_steps, total_steps)

        # Validation phase
        if epoch % config['validation']['interval'] == 0:
            val_loss, metrics = validate(model, val_loader, device, config, writer, epoch)
            
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
    scheduler_type = config["scheduler"]["type"]
    
    if scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Assuming you monitor validation SDR
            factor=config["scheduler"].get("factor", 0.5),
            patience=config["scheduler"].get("patience", 3)
        )
    elif scheduler_type == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["scheduler"].get("T_max", 10)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

def train_epoch(model, loader, optimizer, scaler, device, config, grad_accum, writer, epoch, max_steps=None, total_steps=0):
    train_loss = 0.0
    optimizer.zero_grad()
    steps_completed = 0
    
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
            free_memory()

        train_loss += loss.item()
        log_training(writer, epoch*len(loader)+batch_idx, loss.item())
 
        steps_completed += 1
        
    return train_loss / len(loader), steps_completed

def validate(model, loader, device, config, writer, epoch):
    model.eval()
    val_loss = 0.0
    metrics = {'sdr': [], 'sir': [], 'sar': []}

    with torch.no_grad():
        for mixed, clean in loader:
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, help='Checkpoint path to resume from')
    parser.add_argument('--max_steps', type=int, default=None, help='Limit training to N steps')
    args = parser.parse_args()
    
    train(args.config, args.resume, args.max_steps)
