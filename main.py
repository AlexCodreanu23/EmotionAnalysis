import re
import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords

import pandas as pd
from preprocess import preprocess

nltk.download("stopwords")

nlp = spacy.load("en_core_web_sm")

STOPWORDS = set(stopwords.words("english"))


def normalize_text(text: str) -> str:
    """Lowercase, strip spaces, normalize unicode."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def remove_noise(text: str, remove_emojis=False) -> str:

    text = re.sub(r"http\S+|www\.\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    if remove_emojis:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  
            "\U0001F300-\U0001F5FF"  
            "\U0001F680-\U0001F6FF"  
            "\U0001F1E0-\U0001F1FF"  
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r"", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str):
    doc = nlp(text)
    return [token.text for token in doc]


def lemmatize(text: str):
    doc = nlp(text)
    return [token.lemma_ for token in doc]


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def preprocess(text: str,
               remove_sw=True,
               remove_emojis=False) -> list:
    """
    Full preprocessing pipeline:
    - normalize → noise removal → tokenize → lemmatize → stopword removal
    Returns a list of clean tokens.
    """

    text = normalize_text(text)
    text = remove_noise(text, remove_emojis)

    tokens = lemmatize(text)

    if remove_sw:
        tokens = remove_stopwords(tokens)

    return tokens


if __name__ == "__main__":
    sample = "WOW!! I can't believe this 😱😱! Visit https://news.com <br> This is AMAZING!!!"

    print("Raw:", sample)
    print("Processed:", preprocess(sample))

    df = pd.read_csv("data/sentimentdataset.csv")

    df["processed"] = df["Text"].apply(lambda x: preprocess(str(x)))

    print(df[["Text", "processed"]].head())

    df.to_csv("sentimentdataset_processed.csv", index=False)

    print("Finished! Saved as sentimentdataset_processed.csv")
