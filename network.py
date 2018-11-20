import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Average, Embedding, RepeatVector
from keras.layers import Dot
from keras.models import Model
from data_preprocessing import yield_data, get_emb_idx, create_emb_matrix
from keras.backend import mean, expand_dims
import pickle

batch_size = 1
ip_dim_art = 100
ip_dim_clm = 200
attn_dim_wt = 3
lstm_op_dim = 64
op_1 = 32
op_2 = 1
vocabulary_size = 50000

file_path='snopes.npy'
with open(file_path, 'rb') as f:
    record = pickle.load(f)
emb_idx = get_emb_idx('glove.6B.100d.txt')
word_emb = create_emb_matrix(vocabulary_size, emb_idx, record['claim_text'] + record['evidence'])# + record['evidence_source'])



art_wrd = Input(shape=(100,))
clm_wrd = Input(shape=(100,))
clm_wrd_emb = Embedding(vocabulary_size, 100, input_length=100, weights=[word_emb], trainable=False)(clm_wrd)
mean_clm_wrd_emb = RepeatVector(100)(mean(clm_wrd_emb, axis=-1))
art_wrd_emb = Embedding(vocabulary_size, 100, input_length=100, weights=[word_emb], trainable=False)(art_wrd)
ip_to_dense = Concatenate(axis=-1)([mean_clm_wrd_emb, art_wrd_emb])
attn_weights = Dense(128, activation='tanh')(clm_wrd_emb)
attn_weights = Activation('softmax')(attn_weights)
model_attn = Model(inputs=[art_wrd, clm_wrd], outputs=attn_weights)


lstm_op = Bidirectional(LSTM(lstm_op_dim, return_sequences=True), merge_mode='concat')(art_wrd_emb)
model_lstm = Model(inputs=art_wrd, outputs=lstm_op)


inner_pdt = Dot(axes=1)([model_attn.output, model_lstm.output])
avg = RepeatVector(1)(mean(inner_pdt, axis=-1))[:,0,:]


clm_src = Input(shape=(8,))
art_src = Input(shape=(8,))

final_feat = Concatenate(axis=1)([clm_src, avg, art_src])

x = Dense(op_1)(final_feat)
x = Dense(op_2)(x)
# cred_score = Activation('softmax')(x)

model = Model(inputs=final_feat, outputs=x)

model.fit()

for art_wrd, clm_wrd, art_src, label in yield_data(record):
    clm_src = np.zeros_like(art_src)
    loss = model.train_on_batch(x=[art_wrd, clm_wrd, art_src, clm_src], y=label)