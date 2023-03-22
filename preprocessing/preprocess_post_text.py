import pandas as pd
import re
from gensim.models import Word2Vec

def clean_text(string: str):
    string = string.lower()

    # Remove all special characters
    string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
    # Remove all newlines
    string = re.sub(r'\n', '', string)
    # Remove all links
    string = re.sub(r'http\S+', '', string)
    # Remove all emojis
    emojis_regex = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)
    string = emojis_regex.sub("", string)
    # Remove all extra whitespace
    string = re.sub(r'\s+', ' ', string).strip()

    return string


def process_posts():
    # Reads the CSV containing all post text descriptions
    post_text = pd.read_csv("data/all_posts_metadata_matched.csv")["description"]

    # Cleans all of the enc
    posts = post_text.apply(lambda desc: clean_text(desc).split() if type(desc) == str else desc).dropna().tolist()

    # Generates word embeddings
    model = Word2Vec(posts, min_count=1)
    # Saves the word embedding vectors - the .npy file generated can be used in the MVAE model
    model.wv.save("preprocessing-output/text_embeddings.wordvectors")

if __name__ == "__main__":
    process_posts()