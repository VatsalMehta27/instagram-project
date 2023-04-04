from typing import List
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))


def clean_text(string: str) -> str:
    string = string.lower()

    # Replaces all contractions
    string = re.sub(r"\'s", " is ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " had ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"n\'t", " not", string)

    # Remove all special characters
    string = re.sub(r"[^a-zA-Z\s]", "", string)
    # Remove all newlines
    string = re.sub(r"\n", "", string)
    # Remove all links
    string = re.sub(r"http\S+", "", string)
    # Remove all emojis
    emojis_regex = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )
    string = emojis_regex.sub("", string)
    # Remove all extra whitespace
    string = re.sub(r"\s+", " ", string).strip()

    string = [word for word in string.split() if word not in STOPWORDS]

    return list(string)


# def process_posts(descriptions: List[str]) -> List[str]:

#     # Cleans all of the enc
#     clean_descriptions = [
#         clean_text(text) for text in descriptions if type(text) == str
#     ]

#     return clean_descriptions
