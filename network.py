import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Average
from keras.layers import Dot
from keras.models import Model
from data_preprocessing import yield_data
import tensorflow as tf
from keras.backend import mean, expand_dims

batch_size = 1
ip_dim_art = 100
ip_dim_clm = 200
attn_dim_wt = 3
lstm_op_dim = 64
op_1 = 32
op_2 = 1

art_wrd = Input(shape=(80, 100))
clm_wrd = Input(shape=(80, 200))
attn_weights = Dense(100, activation='tanh')(clm_wrd)
attn_weights = Activation('softmax')(attn_weights)

lstm_op = Bidirectional(LSTM(lstm_op_dim, return_sequences=True), merge_mode='concat')(art_wrd)

inner_pdt = Dot(axes=1)([lstm_op, attn_weights])
avg = expand_dims(mean(inner_pdt, axis=1), axis=1)

clm_src = Input(shape=(4, 100))
art_src = Input(shape=(4, 100))

final_feat = Concatenate(axis=1)([clm_src, avg, art_src])

x = Dense(op_1)(final_feat)
x = Dense(op_2)(x)
cred_score = Activation('softmax')(x)

model = Model(inputs=[art_wrd, clm_wrd, art_src, clm_src], outputs=cred_score)

model.fit()

for art_wrd, clm_wrd, art_src, label in yield_data(file_path='snopes.npy'):
    clm_src = np.zeros_like(art_src)
    loss = model.train_on_batch(x=[art_wrd, clm_wrd, art_src, clm_src], y=label)