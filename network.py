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

batch_size = 32
ip_dim = 3
attn_dim_wt = 3
lstm_op_dim = 64
op_1 = 32
op_2 = 1

model = Sequential()

art_wrd = Input(shape=(ip_dim,))
clm_wrd = Input(shape=(ip_dim,))
avg_clm_wrd = Average()(clm_wrd)
new_clm_wrd = Concatenate()([art_wrd, avg_clm_wrd])
attn_weights = Dense(attn_dim_wt, activation='tanh')(new_clm_wrd)
attn_weights = Activation('softmax')(attn_weights)

lstm_op = Bidirectional(LSTM(lstm_op_dim, return_sequences=True), input_shape=(5, 10))

inner_pdt = Dot(axes=-1)([lstm_op, attn_weights])
avg = np.mean(inner_pdt, axis=0)

clm_src = Input(shape=(ip_dim,))
art_src = Input(shape=(ip_dim,))

final_feat = Concatenate()([clm_src, avg, art_src])

x = Dense(op_1)(final_feat)
x = Dense(op_2)(x)
cred_score = Activation('softmax')(x)

model = Model(inputs=[art_wrd, clm_wrd, art_src, clm_src], outputs=cred_score)

