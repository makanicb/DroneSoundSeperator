# src/evaluate.py
from utils import compute_sdr_sir_sar

# load model & test loader...
for noisy, clean in test_loader:
    mask = model(stft(noisy).to(device))
    est = istft(mask * stft(noisy).abs(), stft(noisy).angle())
    sdr, sir, sar = compute_sdr_sir_sar(clean.numpy(), est.cpu().detach().numpy())
    # aggregate metrics...

