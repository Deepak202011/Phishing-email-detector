# model.py — load model and predict only

import pickle
import numpy as np
from app.config import MODEL_PATH, TFIDF_PATH, PREDICTION_THRESHOLD

class PhishingModel:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.is_loaded = False

    def load(self):
        """Load pkl files from disk"""
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)

            with open(TFIDF_PATH, 'rb') as f:
                self.tfidf = pickle.load(f)

            self.is_loaded = True
            print("Model loaded successfully")

        except Exception as e:
            self.is_loaded = False
            print(f"Model loading failed: {e}")

    def predict(self, features) -> dict:
        """Take features → return prediction"""
        if not self.is_loaded:
            raise Exception("Model not loaded")

        # Get probability
        probability = self.model.predict_proba(features)[0][1]

        # Apply threshold
        is_spam = probability > PREDICTION_THRESHOLD

        return {
            "is_spam": bool(is_spam),
            "prediction": "Spam" if is_spam else "Not Spam",
            "confidence": round(float(probability), 4),
            "message": "Phishing email detected!" if is_spam else "Email looks safe"
        }

# Single instance — loaded once, reused forever
phishing_model = PhishingModel()