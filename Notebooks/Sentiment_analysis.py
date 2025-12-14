import os
import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Resolve path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../Dataset/IMDB Dataset.csv")

df = pd.read_csv(DATASET_PATH)

print(df.head())
print(df.info())
print(df["sentiment"].value_counts())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

df["clean_review"] = df["review"].apply(clean_text)

print(df[["review", "clean_review"]].head())

df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

print(df[["sentiment", "label"]].head())


#machine learning splitting into training-testing datasets

X = df["clean_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,      # strong baseline
    ngram_range=(1, 2)      # unigrams + bigrams = smarter model
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("TF-IDF train shape:", X_train_vec.shape)
print("TF-IDF test shape:", X_test_vec.shape)


model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

model.fit(X_train_vec, y_train)

print("Model training completed.")
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved.")


y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
