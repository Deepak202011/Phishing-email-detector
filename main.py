# main.py — routes only

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schema import EmailInput, PredictionOutput, HealthCheck
from app.model import phishing_model
from app.preprocessor import build_features

# Load model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up — loading model...")
    phishing_model.load()
    yield
    print("Shutting down...")

# Create app
app = FastAPI(
    title="Phishing Email Detector",
    description="Detects phishing emails using ML",
    version="1.0.0",
    lifespan=lifespan
)

# Health check route
@app.get("/", response_model=HealthCheck)
def health_check():
    return {
        "status": "ok",
        "model_loaded": phishing_model.is_loaded
    }

# Prediction route
@app.post("/predict", response_model=PredictionOutput)
def predict(email: EmailInput):
    # Check model loaded
    if not phishing_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # Build features
        features = build_features(email, phishing_model.tfidf)

        # Predict
        result = phishing_model.predict(features)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )