# Drone Sound Separation Project Plan (v6)
# Last Updated: 2025-05-20

## CURRENT STATUS
- Core pipeline operational
- 16-channel audio processing implemented
- API endpoint functional for waveform input/output

## COMPLETED MILESTONES
1. MULTI-CHANNEL INFRASTRUCTURE
   - 16-channel audio validation in API
   - Channel-wise STFT/iSTFT processing
   - Waveform input pipeline (shape: [batch=1, channels=16, samples])
   
2. MODEL ARCHITECTURE
   - UNet modification for raw waveform input
   - Integrated STFT/iSTFT in forward pass
   - Channel-preserving mask estimation

3. API IMPLEMENTATION
   - File upload validation system
   - Error handling framework
   - Direct waveform-to-waveform processing

## REMAINING TASKS
4. VALIDATION & TUNING
   - [ ] Per-channel SDR/SIR/SAR metrics
   - [ ] Real-world noise profile analysis
   - [ ] Cross-channel interference tests

5. DEPLOYMENT PREPARATION
   - [ ] Docker image with CUDA 12.2
   - [ ] Load testing suite
   - [ ] API documentation (16-channel spec)

## KEY ARCHITECTURAL DECISIONS
- End-to-End Processing: Model handles STFT internally
- Channel Isolation: Independent processing per channel
- Validation Requirements:
  * 10+ real-world test recordings
  * <2s latency for 5s audio on T4 GPU
  * <5% performance variance across channels

## IMMEDIATE NEXT STEPS
1. Generate validation dataset:
   - Clean reference: 16-channel sine waves
   - Mixed input: Noise + drone samples

2. Implement metrics collection:
   def compute_channel_metrics(clean, estimate):
       return {f"ch{i}": calculate_sdr(clean[:,i], estimate[:,i]) 
               for i in range(16)}

3. Docker configuration:
   FROM nvidia/cuda:12.2.0-base
   RUN pip install torch==2.0.1 torchaudio==2.0.2 fastapi==0.95.0
   EXPOSE 8000
   CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0"]

## RISK MITIGATION STRATEGY
| Risk Category          | Mitigation Approach                 |
|------------------------|-------------------------------------|
| Channel phase mismatch | Inter-channel coherence checks      |
| GPU memory overflow    | Chunked audio processing            | 
| Real-time performance  | Optimized STFT window functions     |
| Validation coverage    | Synthetic + field-recorded datasets |

# END OF PLAN
