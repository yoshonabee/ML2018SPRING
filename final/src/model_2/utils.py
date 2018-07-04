# -*- coding: UTF-8 -*-
from gensim.models import Word2Vec
import numpy as np
import keras as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
def embedding(data, dim):
  emb_model = Word2Vec(data, size=dim, window=5, min_count=1, workers=12, sg=1, seed=0)
  print(emb_model.most_similar("沒"))
  print('embeding dim:', len(emb_model["沒"]))
  print('vocab size:', len(emb_model.wv.vocab))
  return emb_model

# def splitChinese(inputStr):
#   import jieba
#   seg_list = list(jieba.cut(inputStr, cut_all=False, HMM=True))
#   splitString = " ".join(seg_list)
#   return splitString

class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
  return (K.backend.sigmoid(x)*x)
get_custom_objects().update({'swish': Swish(swish)}) 

class hardSwish(Activation):
    def __init__(self, activation, **kwargs):
        super(hardSwish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def hard_swish(x):
  return (K.backend.hard_sigmoid(x)*x)
get_custom_objects().update({'hard_swish': hardSwish(hard_swish)}) 

class Ht(Activation):
    def __init__(self, activation, **kwargs):
        super(Ht, self).__init__(activation, **kwargs)
        self.__name__ = 'ht'

def ht(x):
    return (K.backend.hard_sigmoid(x)*2 - 1)
get_custom_objects().update({'ht': Activation(ht)}) 

def splitall(string, chWordSplit):
  if chWordSplit == True:
    import jieba
    seg_list = list(jieba.cut(string, cut_all=False, HMM=True))
    splitString = " ".join(seg_list)
    return splitString
  else:
    result = ''
    for c in string:
        result += c + ' '
    return result.strip()

def readtest(testdir):
  import csv
  test_x = []
  with open(testdir, newline='') as csvfile:
    raw = csv.reader(csvfile, delimiter='\n', quotechar=',')
    for i, row in enumerate(raw):
      if i == 0:
        continue
      an5 = row[0].split('5:')[-1].replace('\t','')
      row[0] = row[0].split('5:')[0]
      an4 = row[0].split('4:')[-1].replace('\t','')
      row[0] = row[0].split('4:')[0]
      an3 = row[0].split('3:')[-1].replace('\t','')
      row[0] = row[0].split('3:')[0]
      an2 = row[0].split('2:')[-1].replace('\t','')
      row[0] = row[0].split('2:')[0]
      an1 = row[0].split('1:')[-1].replace('\t','')
      row[0] = row[0].split('1:')[0]
      an0 = row[0].split('0:')[-1].replace('\t','')
      row[0] = row[0].split('0:')[0]
      row[0] = row[0].split(',')[1].split('\t')
      # print(row[0],an0,an1,an2,an3,an4,an5)
      test_x.append([row[0],[an0,an1,an2,an3,an4,an5]])  

  return test_x

def idx2weight(word_idx, vec_dim, w2v):
  unknow = 0
  weight_arr = np.zeros((len(word_idx) + 1, vec_dim)) # (all the word number)*(w2v dim)
  get = 0
  for word, i in word_idx.items(): # translate all the (word index) -> w2v
    try:
      weight_arr[i] = w2v.wv[word]
      get += 1
    except:
      unknow +=1
  print('unknow word:', unknow, ', get:', get)
  return weight_arr


def loadtest():
  # --------------data preprocess --------------------------
  # -------------------------test data load---------------
  """ 
  test_x = readtest('./data/testing_data.csv')
  test_tokens = []
  for i, row in enumerate(test_x):
    [test_tokens.append(splitall(x.replace(' ','').lower(), chsplit)) for x in row[0]]
    [test_tokens.append(splitall(x.replace(' ','').lower(), chsplit)) for x in row[1]]
  # print(test_tokens[0])
  # exit()
  """
  '''
  As = [['','',''],['','',''],['','',''],['','',''],['','',''],['','','']] = [Q1,Q2,...,答案0,答案1,答案2,答案3,答案4,答案5,Q1,Q2,...]
  '''
