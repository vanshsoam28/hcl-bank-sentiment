from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

# Ye line sabse important hai - path ka error kabhi nahi ayega
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

class ReviewRequest(BaseModel):
    review: str

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    review = request.review.strip()
    if not review:
        return {"error": "Review cannot be empty"}
    
    features = vectorizer.transform([review])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    return {
        "prediction": str(prediction),
        "confidence": float(max(probabilities))
    }
