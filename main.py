from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import joblib
import numpy as np
import librosa
import io

# ---------------- API Setup ----------------
app = FastAPI()
API_KEY = "sk_test_123456789"

# Load trained model
model = joblib.load("model.pkl")

# ---------------- Request Schema ----------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ---------------- Feature Extraction ----------------
def extract_features_from_base64(audio_bytes):
    audio_io = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_io, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))

    return np.hstack([mfcc, zcr, spectral_centroid, spectral_flatness, pitch])


# ---------------- API Endpoint ----------------
@app.post("/api/voice-detection")
def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    audio_bytes = base64.b64decode(request.audioBase64)

    features = extract_features_from_base64(audio_bytes)
    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features))

    label = "AI_GENERATED" if prediction == 1 else "HUMAN"

    explanation = (
        "Unnatural pitch stability detected"
        if prediction == 1 else
        "Natural voice variation detected"
    )

    return {
        "status": "success",
        "language": request.language,
        "classification": label,
        "confidenceScore": float(confidence),
        "explanation": explanation
    }

