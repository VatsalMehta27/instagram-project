from gensim.models import Word2Vec


def generate_text_embeddings(text, text_embedding_size, save_text_embeddings=False):
    word_vectors = Word2Vec(text, min_count=1, vector_size=text_embedding_size).wv

    if save_text_embeddings:
        word_vectors.wv.save("preprocessing-output/text_embeddings")

    return word_vectors.vectors, {
        word: index for index, word in enumerate(word_vectors.index_to_key)
    }
