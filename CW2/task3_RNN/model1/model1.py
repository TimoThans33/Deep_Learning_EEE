import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Flatten, Add, Lambda
from keras.layers import LSTM, CuDNNLSTM
from keras.datasets import imdb
from keras.utils import np_utils

# number of most-frequent words to use
nb_words = 5000
n_classes = 1
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=nb_words)
print('x_train:', x_train.shape)
print('x_test:', x_test.shape)
# get_word_index retrieves a mapping word -> index
word_index = imdb.get_word_index()
# We make space for the three special tokens
word_index_c = dict((w, i+3) for (w, i) in word_index.items())
word_index_c['<PAD>'] = 0
word_index_c['<START>'] = 1
word_index_c['<UNK>'] = 2
# Instead of having dictionary word -> index we form
# the dictionary index -> word
index_word = dict((i, w) for (w, i) in word_index_c.items())
# Truncate sentences after this number of words
maxlen = 500
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

## We load the embeddings
## Gensim is a really useful module that provides high-level function
## to use with the embeddings
## Download and unzip GloVe pretrained embeddings
!wget http://nlp.stanford.edu/data/glove.6B.zip
!apt-get -qq install unzip
!unzip glove.6B.zip
!pip install gensim

from gensim.models import word2vec
from gensim.scripts.glove2word2vec import glove2word2vec
## These next three lines just load the embeddings into the object glove_model
glove2word2vec(glove_input_file="glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
from gensim.models.keyedvectors import KeyedVectors
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

import numpy as np
embedding_matrix = np.zeros((nb_words, 300))
for word, i in word_index_c.items():
    if word in glove_model:
      embedding_vector = glove_model[word]
      if embedding_vector is not None and i < nb_words:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector

          
from keras.layers.wrappers import Bidirectional
from keras.layers import GlobalMaxPool1D
## Model parameters:
# Dimensions of the embeddings
embedding_dim = 300

## LSTM dimensionality
lstm_units = 100

print('Build model...')
text_class_model = Sequential()

text_class_model.add(Embedding(nb_words,
                    embedding_dim,
                    input_length=maxlen))
                    weights=[embedding_matrix],
                            trainable=False))

### Do not modify the layers below
text_class_model.add(Bidirectional(CuDNNLSTM(lstm_units, return_sequences = True)))
text_class_model.add(GlobalMaxPool1D())
text_class_model.add(Dense(1, activation='sigmoid'))
text_class_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(text_class_model.summary())

## We train the model for 10 epochs
epochs = 10
validation_split = 0.2
history = text_class_model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_split=validation_split)

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'], label='training')
plt.plot(history.epoch,history.history['val_loss'], label='validation')
plt.title('loss')
plt.legend(loc='best')

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['acc'], label='training')
plt.plot(history.epoch,history.history['val_acc'], label='validation')
plt.title('accuracy')
plt.legend(loc='best');
