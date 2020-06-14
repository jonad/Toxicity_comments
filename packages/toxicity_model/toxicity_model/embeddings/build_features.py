import numpy as np
from keras.preprocessing.text import Tokenizer
from toxicity_model.config import config
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle
import pandas as pd
import os

def build_matrix(word_index, embedding_index):
    embedding_matrix = np.zeros((len(word_index) + 1, config.EMBEDDING_SIZE))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
        except:
            pass
    return embedding_matrix, unknown_words


def get_glove():
    glove2word2vec(os.path.join(config.EMBEDDING_DIR, config.GLOVE_EMBEDDING_FILE),
                   os.path.join(config.EMBEDDING_DIR, config.WORD2VEC_GLOVE_EMBEDDING_FILE))
    glove_model = KeyedVectors.load_word2vec_format(
        os.path.join(config.EMBEDDING_DIR, config.WORD2VEC_GLOVE_EMBEDDING_FILE))
    
    return glove_model


def build_embeddings():
    data = pd.read_csv(os.path.join(config.DATASETS_DIR , config.PROCESSED_DATA_FILE), keep_default_na=False)
    glove_model = get_glove()
    tokenizer = Tokenizer(num_words=500000, filters='', lower=False, oov_token=config.OOV_TOKEN)
    tokenizer.fit_on_texts(list(data[config.TEXT]))
    embedding_matrix, _ = build_matrix(tokenizer.word_index, glove_model)
    print('building embedding')
    # serialize the embedding
    
    with open(os.path.join(config.EMBEDDING_DIR, config.GLOVE_EMBEDDING_MODEL_FILE), 'wb') as handle:
        pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('building the tokenizer')
    with open(os.path.join(config.TOKENIZER_DIR, config.TOKENIZER_FILE), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
if __name__ == "__main__":
    build_embeddings()