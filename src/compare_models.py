import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("data/cleaned_reviews.csv")

X = df["clean_review"]
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


# Logistic Regression
print("\nLOGISTIC REGRESSION")

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr.fit(X_train_tfidf, y_train)

lr_pred = lr.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, lr_pred))

print(
    classification_report(
        y_test,
        lr_pred,
        zero_division=0
    )
)


# Linear SVM
print("\nLINEAR SVM")

svm = LinearSVC(
    class_weight="balanced"
)

svm.fit(X_train_tfidf, y_train)

svm_pred = svm.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, svm_pred))

print(
    classification_report(
        y_test,
        svm_pred,
        zero_division=0
    )
)


print("=" * 60)
print("COMPARISON COMPLETED")
print("=" * 60)
