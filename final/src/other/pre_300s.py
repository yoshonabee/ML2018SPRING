import keras.backend as K
import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, GRU, Embedding, Bidirectional, BatchNormalization, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping
from keras.layers.merge import add, dot, concatenate
import numpy as np
import random
import pandas as pd
import pickle
import os, sys
from gensim.models import Word2Vec
# ------------------uilt ----------
from utils_300s import *
# -------------------------------

vec_dim = 300
chsplit = False

with open('./other/tknzr_300s.pickle', 'rb') as tknfile:
    tknzr = pickle.load(tknfile)

word_idx = tknzr.word_index
wordNum = len(word_idx)
print('got tokens:', wordNum)

if os.path.exists('./other/w2v_300s'):
  w2v = Word2Vec.load('./other/w2v_300s')
else:
  print('./other/w2v_300s not exist')
  exit()
embedding_weights = idx2weight(word_idx, vec_dim, w2v)

longestSent = 17
print("longest sentence:", longestSent)


test_x = readtest(sys.argv[1])
for i, row in enumerate(test_x):
  row[0] = [splitall(x.replace(' ','').lower(), chsplit) for x in row[0]]
  row[0] = pad_sequences(tknzr.texts_to_sequences(row[0]), maxlen=longestSent) # to seq
  row[0] = [np.repeat(sent.reshape(-1, longestSent), 6, axis=0) for sent in row[0]]

  
  y_tensor = []
  for y in row[1]:
    y = splitall(y.replace(' ','').lower(), chsplit)
    y_tensor.append(y)
  y_tensor = pad_sequences(tknzr.texts_to_sequences(y_tensor), maxlen=longestSent) # to seq
  row[1] = np.array(y_tensor)

'''
結果test_x:
dtype: list
每ROW元素：[Qs, As] = [問題(list), 答案(list)]
Qs = [ndarray, ndarray, ...]
As = [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray] = [答案0,答案1,答案2,答案3,答案4,答案5]
'''
model = load_model(sys.argv[3])
model.summary()
embedding_weights = idx2weight(word_idx, vec_dim, w2v)

result = []
for j, row in enumerate(test_x):
  subq = np.array([0,0,0,0,0,0], dtype = 'float64')
  if j % 100 == 0:
    print(j, 'row pre dicted')
  
  for q in row[0]:
    prediction = model.predict([q, row[1]], batch_size=625)
    subq += np.ndarray.flatten(prediction)
  result.append(subq)

print(result)
sample_submit = pd.read_csv("./submission_example.csv")
pred_y = np.argmax(np.array(result).reshape(-1,6),axis=1)
sample_submit["ans"] = pred_y
sample_submit.to_csv(sys.argv[2],index=None)

sample_submit_raw = pd.read_csv("./submission_example.raw.csv")
pred_y = np.array(result)
for i in range(6):
  sample_submit_raw["ans"+str(i)] = pred_y[:,i]
sample_submit_raw.to_csv(sys.argv[2].replace('.csv','.raw.csv'),index=None)
