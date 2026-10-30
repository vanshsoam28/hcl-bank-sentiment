import re
import pandas as pd

INPUT_FILE = "data/bank_review3.csv"
OUTPUT_FILE = "data/cleaned_reviews.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def rating_to_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


print("=" * 60)
print("BANK CUSTOMER REVIEW SENTIMENT ANALYZER")
print("=" * 60)

print("\n1. Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

before = len(df)

df = df.drop_duplicates(subset=["review"])

after = len(df)

print("Duplicate rows removed:", before - after)

df = df.dropna(subset=["review"])

print("\n2. Cleaning reviews...")

df["clean_review"] = df["review"].apply(clean_text)

df = df[df["clean_review"].str.len() > 0]

print("Clean dataset shape:", df.shape)

print("\n3. Creating sentiment labels...")

df["sentiment"] = df["rating"].apply(rating_to_sentiment)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

df.to_csv(OUTPUT_FILE, index=False)

print("\n4. Cleaned dataset saved to:")
print(OUTPUT_FILE)

print("\nPreparation completed successfully!")
