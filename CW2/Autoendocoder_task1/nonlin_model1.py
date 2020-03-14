import numpy as np
import keras
np.random.seed(123)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from keras.datasets import mnist

num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

x_train= x_train.astype('float32')
x_test= x_test.astype('float32')

x_train /= 255.
x_test /= 255.
### Modify the model here, this is an example of a non-linear model
model = Sequential()
# Encoder starts
model.add(Dense(128, activation='relu', input_shape=(784, )))
model.add(Dense(2, name='representation'))
#model.add(Dense(128, activation='relu', input_shape=(784,)))
#model.add(Dense(128))
#model.add(Dense(2, name='representation'))
## And now the decoder
model.add(Dense(784, activation = 'sigmoid'))
#model.add(Dense(784, activation='sigmoid'))
"""
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(2, name='representation'))
model.add(Dense(784, activation='sigmoid'))
"""
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])
epochs = 10
history = model.fit(x_train, x_train, batch_size=128,
          epochs=epochs, validation_split=0.2)
print(model.summary())

print('Test MSE obtained is {:.4f}'.format(model.evaluate(x_test, x_test)[0]))

ind = np.random.randint(x_test.shape[0] -  1)
## The function below is defined in the tutorial
plot_recons_original(x_test[ind], y_test[ind])

## The function below is defined in the tutorial

representation = predict_representation(model, x_test)

## If your autoencoder contains a representation layer 
## with more than 2 dimensions, you can project the representations to a lower
## dimensionality space first by using dimensionality reduction techniques such 
## as PCA or TSNE. TSNE is slower, but results in quite nice visualizations,
## where the different classes form quite distinctive clusters
## TSNE: https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
## TSNE code (~80 seconds)
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# representation_tsne = tsne.fit_transform(representation)

## PCA code
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca = pca.fit(predict_representation(model, x_train))
representation_pca = pca.transform(representation)

## The function below is defined in the tutorial
plot_representation_label(representation, y_test)
