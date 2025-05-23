from fastapi import FastAPI, File, UploadFile, HTTPException 
from fastapi.responses import FileResponse
import torchaudio
import torch
import os
import yaml
from src.inference.model_loader import load_model
from src.utils import stft, istft
from src.inference.schemas import SeparationResponse, HealthCheck

# Load config
with open("configs/inference.yaml") as f:
    config = yaml.safe_load(f)

app = FastAPI(title="Drone Sound Separator API")
model = load_model("experiments/run1/ckpt_best.pt", device=config["device"])

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {"status": "OK"}

@app.post("/separate", response_model=SeparationResponse)
async def separate_audio(file_upload: UploadFile = File(...)):
    try:
        # Validate file extension
        if not file_upload.filename.lower().endswith(tuple(config["allowed_extensions"])):
            raise HTTPException(400, "Invalid file format")

        # Create temp dir
        os.makedirs(config["temp_dir"], exist_ok=True)
        
        # Process file
        input_path = os.path.join(config["temp_dir"], file_upload.filename)
        output_path = os.path.join(config["temp_dir"], f"clean_{file_upload.filename}")
        
        # Save uploaded file
        with open(input_path, "wb") as f:
            content = await file_upload.read()
            if len(content) > config["max_file_size"] * 1024 * 1024:
                raise HTTPException(413, "File too large")
            f.write(content)
        
        # Process audio
        waveform, sr = torchaudio.load(input_path)
        print("Input shape:", waveform.unsqueeze(0).shape)
        spec = stft(waveform.unsqueeze(0))  # Add batch dimension
        print("STFT shape:", spec.shape)
        with torch.no_grad():
            mask = model(spec)
        clean_spec = spec * mask
        clean_wav = istft(clean_spec).squeeze(0)  # Remove batch dimension
        
        # Save result
        torchaudio.save(output_path, clean_wav, sr)
        
        return FileResponse(
           output_path,
           media_type="audio/wav",
           filename=os.path.basename(output_path)
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "success": False,
            "message": "Processing failed",
            "processing_time_ms": 0,
            "error": str(e)
        }
