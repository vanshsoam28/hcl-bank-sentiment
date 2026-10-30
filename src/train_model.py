import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


INPUT_FILE = "data/cleaned_reviews.csv"

MODEL_FILE = "models/sentiment_model.pkl"
VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"


print("=" * 60)
print("BANK CUSTOMER REVIEW SENTIMENT MODEL")
print("=" * 60)


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

print("\n1. Loading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Select features and target
# --------------------------------------------------

X = df["clean_review"]

y = df["sentiment"]


print("\nSentiment distribution:")
print(y.value_counts())


# --------------------------------------------------
# 3. Train/Test Split
# --------------------------------------------------

print("\n2. Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------------
# 4. TF-IDF
# --------------------------------------------------

print("\n3. Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)


X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print(
    "Training feature shape:",
    X_train_tfidf.shape
)

print(
    "Testing feature shape:",
    X_test_tfidf.shape
)


# --------------------------------------------------
# 5. Logistic Regression
# --------------------------------------------------

print("\n4. Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train_tfidf,
    y_train
)


print("Model training completed.")


# --------------------------------------------------
# 6. Prediction
# --------------------------------------------------

print("\n5. Evaluating model...")

predictions = model.predict(
    X_test_tfidf
)


# --------------------------------------------------
# 7. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# --------------------------------------------------
# 8. Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# 9. Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)


print("\n6. Model saved successfully.")

print("Model:")
print(MODEL_FILE)

print("\nVectorizer:")
print(VECTORIZER_FILE)


print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)
