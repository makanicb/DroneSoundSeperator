# tests/generate_test_data.py
import numpy as np
import json
from pathlib import Path
import scipy.signal
import datetime

# Configuration
SAMPLE_RATE = 16000
DURATION = 3.0
CHANNELS = 16
OUTPUT_DIR = Path("tests/test_data")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_drone_audio():
    """Generate synthetic drone audio with harmonics"""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    # Fundamental + harmonics
    fundamental = 120  # Hz (typical drone frequency)
    signal = 0.5 * np.sin(2 * np.pi * fundamental * t)
    signal += 0.2 * np.sin(2 * np.pi * 3 * fundamental * t)  # 3rd harmonic
    signal += 0.1 * np.sin(2 * np.pi * 5 * fundamental * t)  # 5th harmonic
    
    # Add propeller noise (broadband)
    rng = np.random.RandomState(42)
    noise = 0.05 * rng.randn(len(t))
    signal += noise
    
    # Create multichannel version with slight variations
    multichannel = np.zeros((len(t), CHANNELS))
    for ch in range(CHANNELS):
        multichannel[:, ch] = signal * (0.9 + 0.1 * ch/CHANNELS)  # Channel gain variation
    
    # Normalize to -0.8 dBFS peak
    multichannel *= 0.8 / np.max(np.abs(multichannel))
    return multichannel.astype(np.float32)

def generate_noise_audio():
    """Generate realistic environmental noise"""
    rng = np.random.RandomState(42)
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    # Base noise (band-limited)
    noise = rng.randn(len(t))
    
    # Adjust frequency range to stay under Nyquist
    lowcut = 100  # Hz
    highcut = 7900  # Changed from 8000 to 7900 to stay under Nyquist
    b, a = scipy.signal.butter(4, [lowcut, highcut], btype='bandpass', fs=SAMPLE_RATE)
    noise = scipy.signal.lfilter(b, a, noise)
    
    # Add occasional impulses (bird chirps)
    for _ in range(5):
        idx = rng.randint(0, len(t)-100)  # Ensure we don't exceed array bounds
        noise[idx:idx+100] += 0.3 * np.hanning(100)
    
    # Create multichannel version
    multichannel = np.zeros((len(t), CHANNELS))
    for ch in range(CHANNELS):
        multichannel[:, ch] = noise * (0.7 + 0.1 * ch/CHANNELS)
    
    # Normalize to -0.6 dBFS peak
    multichannel *= 0.6 / np.max(np.abs(multichannel))
    return multichannel.astype(np.float32)

def generate_metadata(clean_path, noise_path):
    metadata = {
        "test_files": {
            "clean": {
                "file_path": str(clean_path),
                "description": "Synthetic drone audio with harmonics",
                "specs": {
                    "duration_sec": DURATION,
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                    "drone_info": {
                        "model": "TEST_DJI_Phantom4",
                        "distance_m": 8.0,
                        "propeller_rpm": 2400,
                        "fundamental_hz": 120,
                        "harmonics": [360, 600]
                    },
                    "normalization": {
                        "peak_amplitude": 0.8,
                        "dynamic_range_db": 42.0
                    }
                }
            },
            "noise": {
                "file_path": str(noise_path),
                "description": "Synthetic urban noise with impulses",
                "specs": {
                    "duration_sec": DURATION,
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                    "noise_type": "urban_background",
                    "sources": ["wind", "traffic", "birds"],
                    "frequency_range_hz": [100, 7900],  # Updated to match code
                    "normalization": {
                        "peak_amplitude": 0.6,
                        "dynamic_range_db": 38.0
                    }
                }
            }
        },
        "recording_setup": {
            "microphone_array": "TEST_UMA-16V2",
            "layout": "circular_15cm_diameter",
            "adc_resolution": "24bit",
            "gain_db": 32.0
        },
        "test_purposes": {
            "unit_tests": ["data_loading", "stft_processing", "normalization"],
            "integration_tests": ["mixing", "mask_prediction", "separation_quality"]
        },
        "generation_parameters": {
            "random_seed": 42,
            "creation_date": datetime.datetime.now().isoformat()
        }
    }
    return metadata

if __name__ == "__main__":
    # Generate files
    clean_audio = generate_drone_audio()
    noise_audio = generate_noise_audio()
    
    # Save files
    clean_path = OUTPUT_DIR / "clean.npy"
    noise_path = OUTPUT_DIR / "noise.npy"
    metadata_path = OUTPUT_DIR / "metadata.json"
    
    np.save(clean_path, clean_audio)
    np.save(noise_path, noise_audio)
    
    # Generate and save metadata
    metadata = generate_metadata(clean_path, noise_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated test files in {OUTPUT_DIR}")
    print(f"- Clean drone audio: {clean_path}")
    print(f"- Noise audio: {noise_path}")
    print(f"- Metadata: {metadata_path}")
