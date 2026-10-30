import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import RandomOverSampler


print("=" * 60)
print("BALANCED BANK CUSTOMER REVIEW SENTIMENT ANALYZER")
print("=" * 60)

# 1. Load cleaned dataset
print("\n1. Loading cleaned dataset...")

df = pd.read_csv("data/cleaned_reviews.csv")

X = df["clean_review"].fillna("")
y = df["sentiment"]

print("Dataset shape:", df.shape)

print("\nOriginal sentiment distribution:")
print(y.value_counts())


# 2. Train-test split
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


# 3. TF-IDF
print("\n3. Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape :", X_test_tfidf.shape)


# 4. Random oversampling
print("\n4. Balancing training data using Random Oversampling...")

ros = RandomOverSampler(random_state=42)

X_train_balanced, y_train_balanced = ros.fit_resample(
    X_train_tfidf,
    y_train
)

print("\nBalanced training distribution:")
print(pd.Series(y_train_balanced).value_counts())


# 5. Train Logistic Regression
print("\n5. Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X_train_balanced, y_train_balanced)


# 6. Evaluation
print("\n6. Evaluating model...")

y_pred = model.predict(X_test_tfidf)

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


# 7. Save model and vectorizer
print("\n7. Saving model files...")

joblib.dump(model, "models/balanced_sentiment_model.pkl")
joblib.dump(vectorizer, "models/balanced_tfidf_vectorizer.pkl")

print("\nModel saved:")
print("models/balanced_sentiment_model.pkl")

print("Vectorizer saved:")
print("models/balanced_tfidf_vectorizer.pkl")

print("\n" + "=" * 60)
print("BALANCED MODEL TRAINING COMPLETED")
print("=" * 60)
