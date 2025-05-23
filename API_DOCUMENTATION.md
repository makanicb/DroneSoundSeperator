# Drone Sound Separation API Documentation
# Version: 1.0
# Last Updated: 2025-05-20

=====================
= API SPECIFICATION =
=====================

BASE URL: http://api.droneseparator.com/v1

-------------------------
| Authentication        |
-------------------------
Method: API Key
Header: X-API-Key: <your_key>
Note: Contact support@droneseparator.com for API keys

-------------------------
| Endpoints             |
-------------------------

[POST] /separate
Description: Process 16-channel drone audio recording
Content-Type: multipart/form-data

Parameters:
- file_upload (required): 16-channel WAV audio file
- output_format (optional): ["wav", "flac"] (default: wav)

Request Example:
curl -X POST \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file_upload=@recording.wav" \
  -F "output_format=wav" \
  https://api.droneseparator.com/v1/separate \
  --output cleaned_audio.wav

Successful Response:
- Content-Type: audio/wav or audio/flac
- Body: Processed 16-channel audio file

Error Response (JSON):
{
  "error": "invalid_channels",
  "message": "Audio must contain exactly 16 channels",
  "details": {
    "received_channels": 8,
    "required_channels": 16
  }
}

-------------------------

[GET] /health
Description: Service status check

Response (JSON):
{
  "status": "healthy",
  "version": "1.0.3",
  "uptime": "146h22m",
  "gpu_available": true
}

-------------------------
| Audio Specifications  |
-------------------------
- Format: WAV (PCM 16-bit)
- Channels: 16 (fixed)
- Sample Rate: 44.1kHz or 48kHz
- Duration: 1-30 seconds
- Max Size: 50MB

-------------------------
| Rate Limits           |
-------------------------
- Free Tier: 100 requests/hour
- Pro Tier: 1000 requests/minute
- Batch Mode: Contact sales

-------------------------
| Error Codes           |
-------------------------
400: Bad Request (invalid format)
413: Payload Too Large
415: Unsupported Media Type
429: Rate Limit Exceeded
500: Internal Server Error
503: Service Unavailable

-------------------------
| Changelog             |
-------------------------
2025-05-15 v1.0:
- Initial release with 16-channel support
- Basic noise suppression capabilities

-------------------------
| Support (placeholder) |
-------------------------
Contact: support@droneseparator.com
Documentation: https://docs.droneseparator.com
Status Page: https://status.droneseparator.com

# END OF DOCUMENTATION
