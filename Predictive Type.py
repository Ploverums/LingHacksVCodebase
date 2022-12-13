import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
import pickle
import heapq

path = "x/x/x/train.txt"
text = open(path, encoding='utf8').read().lower()

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

su_words = np.unique(words)
su_word_index = dict((c, i) for i, c in enumerate(su_words))

last_words = [ ]
after_words = [ ]
BackWords = 5
for i in range(len(words) - BackWords):
  last_words.append(words[i:i + BackWords])
  after_words.append(words[i + BackWords])
print(last_words[0])
print(after_words[0])  
X = np.zeros((len(last_words), BackWords, len(su_words)), dtype=bool)
Y = np.zeros((len(after_words), len(su_words)), dtype=bool)
for i, each_words in enumerate(last_words):
    for j, each_word in enumerate(each_words):
        X[i, j, su_word_index[each_word]] = 1
    Y[i, su_word_index[after_words[i]]] = 1
    
model = Sequential()
model.add(LSTM(128, input_shape=(BackWords, len(su_words))))
model.add(Dense(len(su_words)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))
model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))

def prepare_input(text):
    x = np.zeros((1, BackWords, len(su_words)))
    for t, word in enumerate(text.split()):
        x[0, t, su_word_index[word]] = 1.
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text, n=3):
    if text == "":
        return("0")
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [su_words[idx] for idx in next_indices]

