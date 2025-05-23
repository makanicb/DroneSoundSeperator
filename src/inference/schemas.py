from fastapi import UploadFile
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"

class SeparationRequest(BaseModel):
    """
    Expected structure for API requests
    """
    file_url: Optional[str] = Field(
        None,
        example="https://example.com/noisy_audio.wav",
        description="Remote URL of audio to process"
    )
    file_upload: Optional[UploadFile] = Field(
        None,
        description="Direct audio file upload"
    )
    output_format: AudioFormat = Field(
        AudioFormat.WAV,
        example="wav",
        description="Format for processed audio"
    )

class SeparationResponse(BaseModel):
    """
    Standardized API response
    """
    success: bool
    message: str
    processing_time_ms: float
    download_url: Optional[str] = Field(
        None,
        example="/results/clean_audio.wav",
        description="URL to download processed audio"
    )
    error: Optional[str] = Field(
        None,
        example="Invalid audio format",
        description="Error details if success=False"
    )

class HealthCheck(BaseModel):
    status: str = Field(..., example="OK")
