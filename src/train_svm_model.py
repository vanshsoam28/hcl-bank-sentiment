import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("=" * 60)
print("BANK CUSTOMER REVIEW SENTIMENT ANALYZER - SVM")
print("=" * 60)

# Load data
df = pd.read_csv("data/cleaned_reviews.csv")

X = df["clean_review"].fillna("")
y = df["sentiment"]

print("\nDataset:", df.shape)
print("\nSentiment distribution:")
print(y.value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# TF-IDF
print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training features:", X_train_tfidf.shape)
print("Testing features :", X_test_tfidf.shape)

# SVM
print("\nTraining Linear SVM...")

model = LinearSVC(
    class_weight="balanced",
    C=1.0,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["Negative", "Neutral", "Positive"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["Negative", "Neutral", "Positive"]
    )
)

# Save
joblib.dump(model, "models/svm_sentiment_model.pkl")
joblib.dump(vectorizer, "models/svm_tfidf_vectorizer.pkl")

print("\nModel saved:")
print("models/svm_sentiment_model.pkl")

print("Vectorizer saved:")
print("models/svm_tfidf_vectorizer.pkl")

print("\nTraining completed successfully!")
