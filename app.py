from fileinput import close
import websocket
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pyfolio
import pandas_ta  
import math
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')


# Download the data from yahoo finance
data = yf.download('AAPL', period="ytd", auto_adjust=True)

# Focus the dataframe onto stock close price data only
data = data['Close'].to_frame()
close_data = data.filter(['Close'])

# MACHINE LEARNING

# Convert dataframe into numpy array
dataset = close_data.values

# Calculate number of rows to train the model
training_data_len = math.ceil( len(dataset) * .8)

# Scale the data from 0 to 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create training set (80% of entire dataset)
train_data = scaled_data[0:training_data_len, :]

# Split the training set
x_train = []
y_train = []

# X is a set of 60 closing prices, Y is the 61st closing price
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])

# Convert the training sets into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the training sets as LSTM network only accepts 3 dimensional arrays (no. of samples, timesteps and features)
# Number of rows, number of timesteps, number of features is just the closing price only so its one
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# LSTM MODEL
model = Sequential()

# Add a layer, give this layer 50 neurons, return sequences true as another LSTM layer is going to be used
# First layer: give the input shape which will be the number of time steps and number of features
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1],1)))

# False as not using anymore LSTM layers
model.add(LSTM(50, return_sequences=False))

# Add a densely connected layer with 25 neurons
model.add(Dense(25))
model.add(Dense(1))

# COMPILE THE MODEL
# optimizer is used to improve upon the loss function, loss: measure how well model did on training
model.compile(optimizer = 'adam', loss='mean_squared_error')

# TRAIN THE MODEL
# batch size: total number of training examples in a batch, epochs: number of iterations when dataset is passed forward
# and backwards through a neural network
model.fit(x_train, y_train, batch_size=1, epochs=1)



# CREATE TESTING DATA SET
# Starting from when y_test hits the first untouched closing price
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len: , :]

# Form sets of 60 closing prices from test data
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values and invert them to real values instead of the 0-1 value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the RMSE
rmse = np.sqrt(np.mean(predictions - y_test)**2)

# Plot graph (debugging) for predictions
train = close_data[:training_data_len]
valid = close_data[training_data_len:]
valid['Predictions'] = predictions
plt.title('Prediction Model')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# TO PREDICT FUTURE PRICES by using most recent 60 days of data
last_60_days = close_data[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)



# Calculation of SMA and EMA
data['SMA30'] = close_data.rolling(30).mean()
data['EWMA30'] = close_data.ewm(span=30).mean()


# Calculate Simple Moving Average with 20 days window
sma = close_data.rolling(window=20).mean()
# Calculate the standar deviation
rstd = close_data.rolling(window=20).std()
# Calculate the upper and lower bollinger bands
data['Upper Band'] = sma + 2 * rstd
data['Lower Band'] = sma - 2 * rstd

# Calculate RSI
data['RSI'] = pandas_ta.rsi(close_data)



# Remove all data points where there is a NA entry
data.dropna(inplace=True)

# Plot the graph (debugging)
data[['Close', 'SMA30', 'EWMA30', 'Upper Band', 'Lower Band']].plot(label='AAPL',figsize=(16, 8))
#plt.show()

# Show all data (debugging)
#print(data)

