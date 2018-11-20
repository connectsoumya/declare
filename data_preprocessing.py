from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.model_selection import StratifiedKFold
from six.moves import cPickle as pickle
import numpy as np
import csv
import re
from nltk import SnowballStemmer
import string
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences


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


# Create a weight matrix for words in training docs
def create_data(vocab_size, text, max_len):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts=text)
    sequences = tokenizer.texts_to_sequences(text)
    data = pad_sequences(sequences, maxlen=max_len)
    return data


def clean_text(text):
    ## Remove puncuation
    text = unicode(text, "utf-8")
    text = text.translate(string.punctuation)
    ## Convert words to lower case and split them
    text = text.lower().split()
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

def read_tsv_file(file_path='snopes.tsv'):
    cred_label = []
    claim_id = []
    claim_text = []
    claim_source = []  # you need to add this depending on the dataset
    evidence = []
    evidence_source = []
    with open(file_path, 'rb') as f:
        tsv_input = csv.reader(f, delimiter='\t')
        for row in tsv_input:
            cred_label.append(row[0])
            claim_id.append(row[1])
            # claim_text.append(clean_text(row[2]))
            claim_text.append(row[2])
            # evidence.append(clean_text(row[3]))
            # evidence_source.append(clean_text(row[4]))
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


def yield_data(record):
    voc = 50000
    # with open(file_path, 'rb') as f:
    #     record = pickle.load(f)
    # emb_idx = get_emb_idx('glove.6B.100d.txt')
    # # for i in range(len(record['claim_text'])):
    # #     claim_word_emb = create_emb_matrix(8, emb_idx, [record['claim_text'][i]])
    # #     claim_word_emb_mean = np.expand_dims(np.mean(claim_word_emb, axis=0), axis=0)
    # #     article_word_emb = create_emb_matrix(voc, emb_idx, [record['evidence'][i]])
    # #     art_source_emb = create_emb_matrix(4, emb_idx, [record['evidence_source'][i]])
    # #     input_to_dense = np.concatenate((article_word_emb, np.repeat(claim_word_emb_mean, voc, axis=0)), axis=1)
    # #     label = 0 if record['cred_label'][i] == 'false' else 1
    # word_emb = create_emb_matrix(voc, emb_idx, record['claim_text'] + record['evidence'])# + record['evidence_source'])
    for i in range(len(record['claim_text'])):
        claim_word_data = create_data(voc, [record['claim_text'][i]], 100)
        claim_word_data_mean = np.expand_dims(np.mean(claim_word_data, axis=0), axis=0)
        article_word_data = create_data(voc, [record['evidence'][i]], 100)
        art_source_data = create_data(voc, [record['evidence_source'][i]], 8)
        input_to_dense = np.concatenate((article_word_data, claim_word_data_mean), axis=1)
        label = [0 if [record['cred_label']][i] == 'false' else 1]
        yield np.expand_dims(article_word_data, axis=0), np.expand_dims(input_to_dense, axis=0), np.expand_dims(art_source_data, axis=0), np.array([label])




if __name__ == '__main__':
    read_tsv_file()
    # yield_data('snopes.npy')
