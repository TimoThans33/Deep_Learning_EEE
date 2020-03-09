import pandas
import matplotlib.pyplot as plt

!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
data = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
plt.plot(data)
plt.xlabel("Months")
plt.ylabel("Passengers")
plt.title("Airline Passengers from January 1949 to December 1960 (12 years)")
plt.show()

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import numpy as np
import math
# convert pandas data frame in numpy array of float.
data_np = data.values.astype("float32")

# normalize data with min max normalization
normalizer = MinMaxScaler(feature_range = (0, 1))
dataset = normalizer.fit_transform(data_np)

# Using 70% of data for training, 30% for test.
TRAINING_PERC = 0.70

train_size = int(len(dataset) * TRAINING_PERC)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Number of samples training set: " + str((len(train))))
print("Number of samples test set: " + str((len(test))))

batch_size = 32
rnn = Sequential()
rnn.add(LSTM(256, input_shape = (window_size, 1)))
#rnn.add(Dense(units=32, activation='relu'))
rnn.add(Dense(units=256, activation='relu'))
rnn.add(Dense(units=128, activation='relu'))
rnn.add(Dense(units=1 ))
rnn.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ['mse'])
rnn.summary()
rnn.fit(train_X, train_Y, epochs=500, batch_size=batch_size, verbose = 0)

mse_train, train_predict = get_predict_and_score(rnn, train_X, train_Y)
mse_test, test_predict = get_predict_and_score(rnn, test_X, test_Y)
print("Traininign data error: %.2f MSE" % mse_train)
print("Test data error: %.2f MSE" % mse_test)

# Training predictions
train_predictions = np.empty_like(dataset)
train_predictions[:, :] = np.nan
train_predictions[window_size:len(train_predict)+window_size, :] = train_predict
# Test predictions
test_predictions = np.empty_like(dataset)
test_predictions[:, :] = np.nan
test_predictions[len(train_predict)+(window_size * 2) + 1:len(dataset) - 1, :] = test_predict

# Create the plot
plt.figure(figsize = (15,5))
plt.plot(normalizer.inverse_transform(dataset), label = "True value")
plt.plot(train_predictions, label = "Training predictions")
plt.plot(test_predictions, label = "Test predictions")
plt.xlabel("Months")
plt.ylabel("1000 member subscriptions")
plt.title("Comparison true vs. predicted in the training and testing set")
plt.legend()
plt.show()
