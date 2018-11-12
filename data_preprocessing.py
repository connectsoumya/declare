from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.model_selection import StratifiedKFold
from six.moves import cPickle as pickle
import numpy as np
import csv


# Get embeddings from GloVe
def get_emb_idx(file_path='glove.6B.100d.txt'):
    embeddings_index = dict()
    f = open(file_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


# Create a weight matrix for words in training docs
def create_emb_matrix(vocab_size, embeddings_index, text):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts=text)
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix


def read_tsv_file(file_path='snopes.tsv'):
    cred_label = []
    claim_id = []
    claim_text = []
    evidence = []
    evidence_source = []
    with open(file_path, 'rb') as f:
        tsv_input = csv.reader(f, delimiter='\t')
        for row in tsv_input:
            cred_label.append(row[0])
            claim_id.append(row[1])
            claim_text.append(row[2])
            evidence.append(row[3])
            evidence_source.append(row[4])
    record = {'cred_label': cred_label,
              'claim_id': claim_id,
              'claim_text': claim_text,
              'evidence': evidence,
              'evidence_source': evidence_source
              }
    with open(file_path[:-4] + '.npy', 'wb') as f:
        pickle.dump(record, f)


def yield_data(file_path='snopes.npy', batch_size=1):
    with open(file_path, 'rb') as f:
        record = pickle.load(f)
    emb_idx = get_emb_idx('glove.6B.100d.txt')
    claim_word_emb = 2

if __name__ == '__main__':
    yield_data('snopes.npy')
