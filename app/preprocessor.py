# preprocessor.py — cleaning and feature engineering only

import re
import numpy as np
import scipy.sparse as sp
from app.config import CATEGORIES

def clean_text(text: str) -> str:
    """Clean raw email text"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text

def encode_category(category: str) -> list:
    """Convert category to dummy variables same way as training"""
    # Training used drop_first=True — drops 'Admin' (first alphabetically)
    # pd.get_dummies sorts alphabetically before dropping
    all_cats_sorted = sorted(CATEGORIES)  # alphabetical sort
    categories_encoded = all_cats_sorted[1:]  # drop first alphabetically
    encoded = [1 if category == cat else 0 for cat in categories_encoded]
    return encoded

def build_features(email_input, tfidf_vectorizer) -> sp.csr_matrix:
    """
    Takes raw input → returns feature vector ready for model
    Same pipeline as Colab training
    """
    # Step 1 — clean text
    cleaned_text = clean_text(email_input.email_content)

    # Step 2 — TF-IDF on cleaned text
    text_features = tfidf_vectorizer.transform([cleaned_text])

    # Step 3 — structured features
    disposable = int(email_input.disposable)
    length = email_input.length
    category_encoded = encode_category(email_input.category)

    # Step 4 — combine structured into array
    struct_features = np.array(
        [[length, disposable] + category_encoded],
        dtype=float
    )

    # Step 5 — combine text + structured
    final_features = sp.hstack([
        text_features,
        sp.csr_matrix(struct_features)
    ])

    return final_features