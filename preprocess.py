
import re
import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

nlp = spacy.load("en_core_web_sm")

STOPWORDS = set(stopwords.words("english"))


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def remove_noise(text: str, remove_emojis: bool = False) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    if remove_emojis:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r"", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list:
    doc = nlp(text)
    return [token.text for token in doc]


def lemmatize(text: str) -> list:
    doc = nlp(text)
    return [token.lemma_ for token in doc]


def remove_stopwords(tokens: list) -> list:
    """Remove English stopwords from token list."""
    return [t for t in tokens if t not in STOPWORDS]


def preprocess(text: str, remove_sw: bool = True, remove_emojis: bool = False) -> list:
    text = normalize_text(text)
    text = remove_noise(text, remove_emojis)
    tokens = lemmatize(text)

    if remove_sw:
        tokens = remove_stopwords(tokens)

    return tokens


def preprocess_to_string(text: str, remove_sw: bool = True, remove_emojis: bool = False) -> str:
    tokens = preprocess(text, remove_sw, remove_emojis)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "WOW!! I can't believe this 😱😱! Visit https://news.com <br> This is AMAZING!!!"

    print("Raw:", sample)
    print("Processed tokens:", preprocess(sample))
    print("Processed string:", preprocess_to_string(sample))