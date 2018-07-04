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
from utils import *
# -------------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

vec_dim = 100
chsplit = False

with open('./tknzr.pickle', 'rb') as tknfile:
    tknzr = pickle.load(tknfile)
# train_caption_sequences = tknzr.texts_to_sequences(train_caption)
# max_length = np.max([len(i) for i in train_caption_sequences])

# word_seq = tknzr.texts_to_sequences(rawX)
word_idx = tknzr.word_index
# for row in Xlist:
#     for oo in row:
#         if oo.count('e') > 0:
#             print(oo)
# exit()
# print(word_idx['東邊'])
#----------------------------------------------
# print('rawX[0]:', rawX[0])
# print('word_seq[0]:', word_seq[0])
# print('word_idx["沒辦法"]:', word_idx["沒辦法"])
wordNum = len(word_idx)
print('got tokens:', wordNum)

if os.path.exists('./w2v'):
  w2v = Word2Vec.load('./w2v')
else:
  print('./w2v not exist')
  exit()
embedding_weights = idx2weight(word_idx, vec_dim, w2v)

# longestSent = 0
# for i in word_seq:
#   if len(i) > longestSent:
#     longestSent = len(i)
longestSent = 21
print("longest sentence:", longestSent)


test_x = readtest('./../../data/testing_data.csv')
for i, row in enumerate(test_x):
  row[0] = [splitall(x.replace(' ','').lower(), chsplit) for x in row[0]]
#   row[0] = [np.repeat(pad_sequences(tknzr.texts_to_sequences(sent), maxlen=longestSent) , 6, axis=0) for sent in row[0]]
  row[0] = pad_sequences(tknzr.texts_to_sequences(row[0]), maxlen=longestSent) # to seq
  row[0] = [np.repeat(sent.reshape(-1, longestSent), 6, axis=0) for sent in row[0]]

#   tensor = []
#   # embed
#   for r in row[0]:
#     tmp = embedding_weights[r[0]].copy()
#     for ele in r:
#       tmp = np.concatenate((tmp,embedding_weights[ele]))
#     tmp = tmp[vec_dim:]
#     tensor.append(tmp)

#   row[0] = tensor 
  
  y_tensor = []
  for y in row[1]:
    y = splitall(y.replace(' ','').lower(), chsplit)
    y_tensor.append(y)
  y_tensor = pad_sequences(tknzr.texts_to_sequences(y_tensor), maxlen=longestSent) # to seq
  row[1] = np.array(y_tensor)
#   y_tensor = []
#   # embed
#   for r in row[1]:
#     tmp = embedding_weights[r[0]].copy()
#     # print(tmp)
#     for ele in r:
#       tmp = np.concatenate((tmp,embedding_weights[ele]))
#     tmp = tmp[vec_dim:]
#     y_tensor.append(tmp)
  
#   row[1] = y_tensor
'''
結果test_x:
dtype: list
每ROW元素：[Qs, As] = [問題(list), 答案(list)]
Qs = [ndarray, ndarray, ...]
As = [ndarray, ndarray, ndarray, ndarray, ndarray, ndarray] = [答案0,答案1,答案2,答案3,答案4,答案5]
'''
# print(len(test_x[0][0]))
# exit()

# # chinese character level tokenizer
# tokenizer = Tokenizer(num_words=None,filters='\n', lower=True, split=" ", char_level=False)
# tokenizer.fit_on_texts(train_caption + [test_corpus])
# print("number of token in caption:", len(tokenizer.word_index))
# inv_map = {v: k for k, v in tokenizer.word_index.items()}

model = load_model(sys.argv[3],custom_objects={'swish':swish, 'ht':ht})
model.summary()
embedding_weights = idx2weight(word_idx, vec_dim, w2v)

result = []
for j, row in enumerate(test_x):
  subq = np.array([0,0,0,0,0,0], dtype = 'float64')
  if j % 100 == 0:
    print(j, 'row pre dicted')
  
  for q in row[0]:
    # print('ans:',kmean.predict(pca.transform(q.reshape(1,-1))))
    # for i in range(6):
    #   # print(kmean.transform(pca.transform(np.array(row[1][i]).reshape(1,-1))))
    #   # # print(kmean.fit_predict(pca.transform(np.array(row[1][i]).reshape(1,-1))))
    #   # # print(kmean.fit_transform(pca.transform(np.array(row[1][i]).reshape(1,-1))))
    #   # print(kmean.predict(pca.transform(np.array(row[1][i]).reshape(1,-1))))
    # #   print(q.shape, row[1][i].shape)
    # #   exit()
    prediction = model.predict([q, row[1]], batch_size=625)
    # print(np.ndarray.flatten(prediction))
    # exit()
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
# # ----------------------------------------------------------
# X = pad_sequences(word_seq, maxlen=longestSent)

# test_caption_sequences =  tknzr.texts_to_sequences([" ".join(sample) for sample in test_sentences])

# # pad sequence
# test_caption_pad = pad_sequences(test_caption_sequences, maxlen=max_length)
# test_data_pad = pad_sequences(test_data, maxlen=max_frame_length,dtype='float32')
# test_data_pad_expand = np.repeat(test_data_pad, 4,axis=0)
# # revert
# print(test_caption_pad.shape)
# print(test_data_pad_expand .shape)

# model_path = "../model"

# model_names = ["model6_randomAug1_1V5_2layers_1024_00.h5"]

# p = []
# for name in model_names:
#     model = load_model(os.path.join(model_path,name))
#     print(name)
#     prediction = model.predict([test_data_pad_expand,test_caption_pad])
#     p.append(prediction)
# pred_y_prob = np.sum(p,axis = 0)
# # load submit
# sample_submit = pd.read_csv("sample_submission.csv")
# pred_y = np.argmax(pred_y_prob.reshape(-1,4),axis=1)
# sample_submit["answer"] = pred_y
# sample_submit.to_csv('./oooo.csv',index=None)
