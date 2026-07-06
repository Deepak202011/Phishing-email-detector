# schema.py — input/output shapes only

from pydantic import BaseModel, Field
from typing import Optional

# What API expects from user
class EmailInput(BaseModel):
    email_content: str = Field(..., min_length=1, description="Email body text")
    domain: str = Field(..., min_length=1, description="Sender domain")
    disposable: bool = Field(..., description="Is disposable email?")
    length: int = Field(..., gt=0, description="Email length")
    category: str = Field(..., description="Email category")

# What API returns back
class PredictionOutput(BaseModel):
    is_spam: bool
    prediction: str
    confidence: float
    message: str

# Health check response
class HealthCheck(BaseModel):
    status: str
    model_loaded: bool