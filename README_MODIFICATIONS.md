# README Modification Checklist

1. PROJECT OVERVIEW
   - Add 16-channel audio processing requirement
   - Clarify end-to-end waveform processing workflow
   - Specify CUDA 12.2+ dependency
   - Add architecture diagram description:
     "16-channel WAV → API → STFT → UNet Mask → iSTFT → Cleaned 16-channel WAV"

2. QUICKSTART GUIDE
   - Update installation commands:
     # Clone and setup
     git clone https://github.com/yourrepo/drone-sound-separation
     conda create -n dss python=3.11
     conda activate dss
     pip install -e ".[dev]"
     
     # Start API
     uvicorn src.inference.app:app --reload

3. API USAGE EXAMPLES
   - Add 16-channel curl example:
     curl -X POST -F "file_upload=@16channel_recording.wav" http://localhost:8000/separate --output cleaned.wav

4. INPUT REQUIREMENTS
   - Add specification table using plain text:
     -----------------------------------------------
     | Parameter       | Requirement               |
     -----------------------------------------------
     | Channels        | 16 (exact)                |
     | Sample Rate     | 44.1kHz or 48kHz          |
     | Bit Depth       | 16-bit PCM                |
     | Duration        | 1-30 seconds              |
     | Max File Size   | 50MB                      |
     -----------------------------------------------

5. TROUBLESHOOTING
   - Common error solutions:
     - "ValueError: too many values to unpack": Verify input audio has exactly 16 channels
     - "400: Invalid channels": Check with soxi <file> to validate channel count
     - CUDA OOM errors: Reduce audio duration below 30 seconds

6. TESTING INSTRUCTIONS
   - Add test generation command:
     python scripts/generate_test_audio.py --channels 16 --duration 5 --output test_16ch.wav
   - Update test command:
     pytest tests/ --verbose --log-level=DEBUG

7. DEPENDENCIES
   - Update requirements section:
     Core Requirements:
     - Python 3.11
     - PyTorch 2.0.1+cu118
     - Torchaudio 2.0.2
     - FastAPI 0.95.0
     - libsndfile1 (system package)

8. CONTRIBUTING GUIDELINES
   - Add channel dimension rules:
     * All audio processing must maintain 16-channel structure
     * Tests must include 16-channel validation cases
     * Pre-commit checks for channel count validation

9. LICENSE & CITATION
   - Update copyright to 2025
   - Add citation boilerplate:
     "If using this work in research, please cite:
     [Pending publication]"

10. SUPPORT RESOURCES
    - Add contact channels:
      Email: audio-support@yourcompany.com
      Discord: https://discord.gg/yourcommunity
      Status Page: https://status.yourcompany.com

# END OF CHECKLIST
