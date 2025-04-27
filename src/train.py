# src/train.py
import yaml, torch, torch.nn as nn, torch.optim as optim
from data_loader import DroneMixtureDataset
from model import UNetSeparator
from utils import stft, istft, si_sdr_loss

def train(cfg):
    # Load config
    with open(cfg) as f: config = yaml.safe_load(f)
    # Dataset & Loader
    ds = DroneMixtureDataset(config["data_dir"], config["snr_levels"])
    loader = torch.utils.data.DataLoader(ds, batch_size=config["batch_size"], shuffle=True)
    # Model, optimizer, loss
    model = UNetSeparator().to(config["device"])
    opt = optim.Adam(model.parameters(), lr=config["lr"])
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        for noisy_wav, clean_wav in loader:
            noisy_wav = noisy_wav.to(config["device"])
            clean_wav = clean_wav.to(config["device"])
            # STFT
            X = stft(noisy_wav)       # (B, 1, F, T)
            S = stft(clean_wav)
            # Predict mask & reconstruct
            M = model(X)
            est_mag = M * X.abs()
            est_wav = istft(est_mag, X.angle())
            # Loss: time-domain SI-SDR
            loss = si_sdr_loss(est_wav, clean_wav)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}: loss={epoch_loss/len(loader):.4f}")
        torch.save(model.state_dict(), f"{config['save_dir']}/ckpt_epoch{epoch}.pt")

