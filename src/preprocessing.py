import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string


def download_nltk_data():
    """Download necessary NLTK data."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def load_imdb():
    """Load IMDb 50K dataset via Hugging Face datasets."""
    from datasets import load_dataset
    print("Loading IMDb dataset...")
    ds = load_dataset("imdb")
    train = pd.DataFrame(ds["train"])
    test  = pd.DataFrame(ds["test"])
    df = pd.concat([train, test], ignore_index=True)
    df = df.rename(columns={"text": "review"})
    df["sentiment"] = df["label"].map({0: "negative", 1: "positive"})
    print(f"  IMDb: {df.shape[0]} samples")
    return df[["review", "sentiment"]]


def load_sst2():
    """
    Load Stanford Sentiment Treebank v2 (SST-2 from GLUE).
    Short single-sentence samples — great for fixing short-text classification.
    Only train split has labels; validation split used as extra test data.
    """
    from datasets import load_dataset
    print("Loading SST-2 dataset...")
    ds = load_dataset("glue", "sst2")
    # train (67k labeled) + validation (872 labeled)
    frames = []
    for split in ("train", "validation"):
        df = pd.DataFrame(ds[split])
        df = df.rename(columns={"sentence": "review"})
        df["sentiment"] = df["label"].map({0: "negative", 1: "positive"})
        frames.append(df[["review", "sentiment"]])
    df = pd.concat(frames, ignore_index=True)
    print(f"  SST-2: {df.shape[0]} samples")
    return df


def load_data():
    """
    Combine IMDb (long reviews) + SST-2 (short sentences).
    Total ~118K samples covering both short and long texts.
    """
    print("=" * 50)
    imdb = load_imdb()
    sst2 = load_sst2()
    df = pd.concat([imdb, sst2], ignore_index=True).dropna(subset=["review", "sentiment"])
    df = df[df["review"].str.strip() != ""]
    print(f"Combined dataset: {df.shape[0]} samples | "
          f"{df['sentiment'].value_counts().to_dict()}")
    print("=" * 50)
    return df


def preprocess_text(text):
    """Lowercase → remove punctuation → remove stopwords."""
    if not isinstance(text, str):
        return str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


def simple_tokenizer(text):
    """Whitespace tokenizer for TfidfVectorizer."""
    return text.split()


if __name__ == "__main__":
    download_nltk_data()
    df = load_data()
    print(df.head())
    sample = "I like this movie."
    print(f"\nOriginal:  {sample}")
    print(f"Processed: {preprocess_text(sample)}")
