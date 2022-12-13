import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

path = 'dev.json'
text = open(path).read().lower()

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
  
X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1
    
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
