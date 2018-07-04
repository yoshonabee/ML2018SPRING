# -*- coding: UTF-8 -*-
import numpy as np
import keras as K
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, GRU, Embedding, Bidirectional, BatchNormalization, TimeDistributed, Activation, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adamax
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping
from keras.layers.merge import add, dot, concatenate
import numpy as np
import random
from gensim.models import Word2Vec
import pandas as pd
import os, sys
import pickle
# ------------------ uilt ----------
from utils import *
# ----------------------------------
from keras.utils.generic_utils import get_custom_objects

output_path = './oo.csv'
chsplit = True
vec_dim = 200
batch_size = 2048
epochs = 100
if len(sys.argv) >= 3:
  continueTrain = sys.argv[2]#'./model_1v11048-0.725-0-0.h5'
else:
  continueTrain = ''
transferNum = 5

rawX = []
y = []
split = []
from os import listdir
traindir = sys.argv[1]
if traindir[-1] != '/':
  traindir + '/'
ld = listdir(traindir)
for i, name in enumerate(ld): # ----from 1~5 file
  # if i != 3:
  with open(sys.argv[1] + name,'r') as trainfile:
    for row in trainfile:
      row = row.replace('\n','').lower()
      ch = splitall(row, chsplit)
      rawX.append(ch)
      y.append(i)
  split.append(len(rawX)-1)

Xlist = [ word.split(" ") for word in rawX]
if os.path.exists('./w2v'):
  w2v = Word2Vec.load('./w2v')
else:
  w2v = embedding(Xlist, vec_dim)# FastText(Xlist, size=300, window=5, min_count=5, workers=12, sg=1, seed=0)
  # w2v = embedding(Xlist + [sent.split(' ') for sent in test_tokens], vec_dim)# FastText(Xlist, size=300, window=5, min_count=5, workers=12, sg=1, seed=0)
  w2v.save('./w2v')
#   print(w2v.wv['iphone'])
# exit()
print('got',len(Xlist),'data.')    
#------------------token----------------
tknzr = Tokenizer(filters='\t')
# print(rawX[0])
# exit()
# tknzr.fit_on_texts(rawX + test_tokens)
tknzr.fit_on_texts(rawX)
with open('./tknzr.pickle', 'wb') as tknfile:
    pickle.dump(tknzr, tknfile)
word_seq = tknzr.texts_to_sequences(rawX)
word_idx = tknzr.word_index
#----------------------------------------------
print('rawX[0]:', rawX[0])
print('word_seq[0]:', word_seq[0])
print('word_idx["沒"]:', word_idx["沒"])
wordNum = len(word_idx)
print('got tokens:', wordNum)
embedding_weights = idx2weight(word_idx, vec_dim, w2v)
longestSent = 0
for i in word_seq:
  if len(i) > longestSent:
    longestSent = len(i)

print("longest sentence:", longestSent)

X = pad_sequences(word_seq, maxlen=longestSent)
y = np.array(X[1:].copy())
X = np.array(X[:len(X)-1])
np.delete(X, split, axis=0)
np.delete(y, split, axis=0)

'''
ex: 
longestSentence = 8
[0,0,0,0,10,218,100,12]
numbers n(int) in list is the sequence in embedding_weights[n(int)]
'''
# -----------------------------------------------
# model

'''def ht(x):
    return (K.backend.hard_sigmoid(x)*2 - 1)
get_custom_objects().update({'ht': Activation(ht)}) 
'''
# ======= define model ======
preInput = Input(shape=(longestSent,), dtype='int32')
em1 = Embedding(len(tknzr.word_index) +1,
                output_dim = vec_dim,
                weights=[embedding_weights],
                input_length=longestSent,
                trainable=False)(preInput)
preSent = Bidirectional(GRU(256, 
                          activation=ht, 
                          dropout=0.2, 
                          recurrent_dropout=0.4,
                          return_sequences=True))(em1)
preSent = Bidirectional(GRU(256, 
                          activation=ht, 
                          recurrent_dropout=0.5,
                          dropout=0.25))(preSent)
# preSent = BatchNormalization()(preSent)
preSent = Dense(512,activation=swish)(preSent)
preSent = Dropout(0.3)(preSent)
preSent = Dense(512,activation=swish)(preSent)
preSent = Dropout(0.4)(preSent)

posInput = Input(shape=(longestSent,), dtype='int32')
em2 = Embedding(len(tknzr.word_index) +1,
                output_dim= vec_dim, 
                weights=[embedding_weights],
                input_length=longestSent,
                trainable=False)(posInput)
posSent = Bidirectional(GRU(256, 
                          activation=ht, 
                          dropout=0.2, 
                          recurrent_dropout=0.4,
                          return_sequences = True))(em2)
posSent = Bidirectional(GRU(256, 
                          activation=ht, 
                          recurrent_dropout=0.5,
                          dropout=0.3))(posSent)
# posSent = BatchNormalization()(posSent)
posSent = Dense(512,activation=swish)(posSent)
preSent = Dropout(0.3)(preSent)
posSent = Dense(512,activation=swish)(posSent)
preSent = Dropout(0.4)(preSent)

merge = K.layers.dot([preSent, posSent],1)
output_dense = Dense(1,activation="sigmoid")(merge)
if continueTrain != '':
  print('load model')
  model = load_model(continueTrain, custom_objects={'swish': swish,'ht':ht})
else:
  model = Model(inputs=[preInput, posInput], outputs=output_dense)
adam = Adamax(lr = 0.0002)
model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['acc'])

print(model.summary())

# print(train_data_pad)
for i in range(0,20):
    # ======= train valid split =======
    train_pre, valid_pre, train_pos, valid_pos = train_test_split(X, y, test_size=0.05)
    for epoch in range(epochs):
        false_pre = []
        false_pos = train_pos.copy()

        true_pre = train_pre
        true_pos = train_pos

        false_pre_tuple = ()
        false_pos_tuple = ()
        
        roll_sample = np.random.choice(len(train_pre), transferNum, replace=False)
        for i in range(transferNum):
          false_pre_tuple += (np.roll(train_pre,roll_sample[i],axis=0),)
          false_pos_tuple += (train_pos,)
        # create transfer learning data of wrong answer pair
        false_pre = np.concatenate(false_pre_tuple) # first sentance
        false_pos = np.concatenate(false_pos_tuple) # response sentance
        # create transfer learning data of right answer pair
        true_pre = train_pre  # first sentance
        true_pos = train_pos  # response sentance
        '''
        set right answer with tag 1, wrong with tag 0
        '''
        ground_truth = [ 1 for _ in range(len(true_pre))] + [0 for _ in range(len(false_pre))]

        # put correct and wrong pre-sentance together
        pre = np.concatenate((true_pre, np.array(false_pre)))
        # put correct and wrong pos-sentance together
        pos = np.concatenate((true_pos, np.array(false_pos)))       

        total_sample_size = len(ground_truth)
        random_index = np.random.choice(total_sample_size,total_sample_size, replace=False)

        pre_input = pre[random_index]
        pos_input = pos[random_index]
        input_ground_truth = np.array(ground_truth)[random_index]

        hist = History()
    
        check_save  = ModelCheckpoint('model_1v' + str(transferNum) + str(batch_size)+ '-{val_acc:.3f}-'+str(i)+'-'+str(epoch)+'.h5', monitor='val_acc', period= 1, save_best_only=True)

        model.fit([pre_input, pos_input], input_ground_truth,
                  batch_size=batch_size,
                  validation_data = ([valid_pre, valid_pos],np.ones(len(valid_pre))),
                  epochs=100, callbacks=[check_save, hist])
