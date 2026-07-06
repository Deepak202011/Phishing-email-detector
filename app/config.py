# config.py — constants and paths only

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

# Model settings
PREDICTION_THRESHOLD = 0.5

# Categories from training data
CATEGORIES = [
    'Admin', 'Billing', 'Contact', 'Feedback', 'General',
    'Info', 'Marketing', 'Owner', 'Sales',
    'Support', 'System'
]