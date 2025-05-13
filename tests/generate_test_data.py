# tests/generate_test_data.py
import numpy as np
import json
from pathlib import Path
import scipy.signal
import datetime

# Configuration - Updated for 44.1kHz
SAMPLE_RATE = 44100
DURATION = 3.0
CHANNELS = 16
SESSION_NAME = "20231001-TEST-44k"  # Matches YYMMDD-TEST format

# Path setup
OUTPUT_DIR = Path("tests/test_data") / SESSION_NAME
AUDIO_CHUNKS_DIR = OUTPUT_DIR / "audio_chunks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_CHUNKS_DIR.mkdir(exist_ok=True)

def generate_drone_audio():
    """Generate 44.1kHz synthetic drone audio with harmonics"""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    # Fundamental + harmonics
    fundamental = 120  # Hz
    signal = 0.5 * np.sin(2 * np.pi * fundamental * t)
    signal += 0.2 * np.sin(2 * np.pi * 3 * fundamental * t)  # 3rd harmonic
    signal += 0.1 * np.sin(2 * np.pi * 5 * fundamental * t)  # 5th harmonic
    
    # Add propeller noise (broadband)
    rng = np.random.RandomState(42)
    noise = 0.05 * rng.randn(len(t))
    signal += noise
    
    # Create multichannel version
    multichannel = np.zeros((len(t), CHANNELS))
    for ch in range(CHANNELS):
        multichannel[:, ch] = signal * (0.9 + 0.1 * ch/CHANNELS)
    
    multichannel *= 0.8 / np.max(np.abs(multichannel))
    return multichannel.astype(np.float32)

def generate_noise_audio():
    """Generate 44.1kHz environmental noise"""
    rng = np.random.RandomState(42)
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    # Broader frequency range for 44.1kHz
    lowcut = 100
    highcut = 20000  # Updated for 44.1kHz Nyquist
    b, a = scipy.signal.butter(4, [lowcut, highcut], btype='bandpass', fs=SAMPLE_RATE)
    noise = scipy.signal.lfilter(b, a, rng.randn(len(t)))
    
    # Add impulses
    for _ in range(5):
        idx = rng.randint(0, len(t)-100)
        noise[idx:idx+100] += 0.3 * np.hanning(100)
    
    # Multichannel version
    multichannel = np.zeros((len(t), CHANNELS))
    for ch in range(CHANNELS):
        multichannel[:, ch] = noise * (0.7 + 0.1 * ch/CHANNELS)
    
    multichannel *= 0.6 / np.max(np.abs(multichannel))
    return multichannel.astype(np.float32)

def generate_metadata():
    """Generate metadata matching project structure"""
    metadata = {
        "session_id": SESSION_NAME,
        "recording_date": datetime.datetime.now().isoformat(),
        "sample_rate": SAMPLE_RATE,
        "recording_setup": {
            "microphone_array": "UMA-16V2",
            "layout": "circular_15cm_diameter",
            "adc_resolution": "24bit",
            "gain_db": 32.0
        },
        "drone_info": {
            "model": "DJI_Phantom4",
            "distance_m": 8.0,
            "propeller_rpm": 2400,
            "fundamental_hz": 120
        },
        "audio_chunks_timestamps": [
            {
                "chunk_index": 0,
                "start_time": 0.0,
                "end_time": 3.0,
                "description": "Clean drone audio with harmonics",
                "specs": {
                    "harmonics": [360, 600],
                    "normalization": {
                        "peak_amplitude": 0.8,
                        "dynamic_range_db": 42.0
                    }
                }
            },
            {
                "chunk_index": 1,
                "start_time": 3.0,
                "end_time": 6.0,
                "description": "Urban noise with impulses",
                "specs": {
                    "noise_type": "urban_background",
                    "sources": ["wind", "traffic", "birds"],
                    "frequency_range_hz": [100, 20000],
                    "normalization": {
                        "peak_amplitude": 0.6,
                        "dynamic_range_db": 38.0
                    }
                }
            }
        ],
        "generation_parameters": {
            "random_seed": 42,
            "creation_date": datetime.datetime.now().isoformat()
        }
    }
    return metadata

if __name__ == "__main__":
    # Generate and save chunks
    np.save(AUDIO_CHUNKS_DIR / "chunk_0.npy", generate_drone_audio())
    np.save(AUDIO_CHUNKS_DIR / "chunk_1.npy", generate_noise_audio())

    # Generate and save metadata
    metadata = generate_metadata()
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated 44.1kHz test data in:\n{OUTPUT_DIR}")
    print("Structure:")
    print(f"├── audio_chunks/")
    print(f"│   ├── chunk_0.npy (clean drone)")
    print(f"│   └── chunk_1.npy (noise)")
    print(f"└── metadata.json")
