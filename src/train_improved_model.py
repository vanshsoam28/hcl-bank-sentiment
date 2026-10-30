import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("=" * 60)
print("IMPROVED BANK CUSTOMER REVIEW SENTIMENT MODEL")
print("=" * 60)

df = pd.read_csv("data/cleaned_reviews.csv")

X = df["clean_review"].fillna("")
y = df["sentiment"]

print("\nDataset:", df.shape)
print("\nSentiment distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nCreating combined TF-IDF features...")

features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            max_features=7000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words="english"
        )
    ),
    (
        "character",
        TfidfVectorizer(
            analyzer="char_wb",
            max_features=5000,
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True
        )
    )
])

X_train_features = features.fit_transform(X_train)
X_test_features = features.transform(X_test)

print("Training feature shape:", X_train_features.shape)
print("Testing feature shape :", X_test_features.shape)

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=1.0,
    random_state=42
)

model.fit(X_train_features, y_train)

y_pred = model.predict(X_test_features)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\nAccuracy: {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))

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

joblib.dump(model, "models/improved_sentiment_model.pkl")
joblib.dump(features, "models/improved_tfidf_vectorizer.pkl")

print("\nSaved:")
print("models/improved_sentiment_model.pkl")
print("models/improved_tfidf_vectorizer.pkl")

print("\nTraining completed successfully!")
