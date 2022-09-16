# Importing libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta
import math
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')


# Read what are the top 500 companies trading currently
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers_list = tickers.Symbol.to_list()

# Three stocks for debugging purposes as time required is shorter
#tickers_list = ['PLTR', 'CHPT', 'SKLZ']

# Initialise lists for storing each ticker's rmse and points
tickers_rmse = {}
tickers_points = {}

for ticker in tickers_list:

    # Download ticker data from yahoo
    data = yf.download(ticker, period="max", auto_adjust=True)

    # Focus the dataframe onto stock close price data only
    data = data['Close'].to_frame()
    close_data = data.filter(['Close'])
    last_price = close_data.values[-1]

    # Initialisation of ticker points and rmse
    tickers_points[ticker] = 0
    tickers_rmse[f'RMSE for {ticker}'] = 0

    # Running the machine learning algo 10 times and taking the average of those outcomes
    for _ in range(10):
    #for _ in range(1): (debugging as runtime is shorter)

        points = 0

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

        # Get the RMSE and update ticker rmse (/10 as loop is done 10 times and average is taken)
        rmse = np.sqrt(np.mean(((predictions - y_test)/y_test)**2))
        tickers_rmse[f'RMSE for {ticker}'] += float(rmse/10)

        # Plot graph (debugging) for predictions
        # train = close_data[:training_data_len]
        # valid = close_data[training_data_len:]
        # valid['Predictions'] = predictions
        # plt.title('Prediction Model')
        # plt.xlabel('Date')
        # plt.ylabel('Close Price')
        # plt.plot(train['Close'])
        # plt.plot(valid[['Close', 'Predictions']])
        # plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        # plt.show()


        # TO PREDICT FUTURE PRICES by using most recent 60 days of data
        last_60_days = close_data[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        pred_change = pred_price - last_price
        
        # If error is too high, reduce weightage of the machine learning prediction
        if rmse>1:
            points += (pred_change/last_price) * 100 * (0.05)
        else:
            # Normalisation of predicted stock price change and taking into account how big the error is
            points += (pred_change/last_price) * 100 * (1-rmse)



        # TECHNICAL INDICATORS

        # Calculation of SMA and EMA
        data['SMA30'] = close_data.rolling(30).mean()
        # If current price is above SMA, it signals upwards momentum
        if data['SMA30'][-1] < last_price:
            points += .5
        elif data['SMA30'][-1] < last_price:
            points -= .5

        data['EWMA30'] = close_data.ewm(span=30).mean()
        if data['EWMA30'][-1] < last_price:
            points += .3
        elif data['EWMA30'][-1] < last_price:
            points -= .3


        # BOLLINGER BANDS

        # Calculate Simple Moving Average with 20 days window
        sma = close_data.rolling(window=20).mean()
        # Calculate the standard deviation
        rstd = close_data.rolling(window=20).std()
        # Calculate the upper and lower bollinger bands
        data['Upper Band'] = sma + 2 * rstd
        data['Lower Band'] = sma - 2 * rstd
        # If price is below bollinger band, it signals upward pressure
        if data['Upper Band'][-1] < last_price:
            points -= 1
        elif data['Lower Band'][-1] > last_price:
            points += 1

        # RSI
        data['RSI'] = pandas_ta.rsi(close = data['Close'])
        last_rsi = data['RSI'][-1]
        # If RSI<30, it is oversold, higher probability that it will rise
        if data['RSI'][-1] < 30:
            points += 1
        elif data['RSI'][-1] < 40:
            points += .5
        elif data['RSI'][-1] > 60:
            points -= .5
        elif data['RSI'][-1] > 70:
            points -= 1

        # FUNDAMENTALS

        # Use yahoo finance to find information about the stock
        ticker_obj = yf.Ticker(ticker)
        price_to_book_ratio = ticker_obj.info['priceToBook']
        # If PB ratio is under a certain threshold, it is undervalued
        if price_to_book_ratio<.9:
            points += .5
        elif price_to_book_ratio>1.1:
            points -= .5
        price_to_earnings_ratio = ticker_obj.info['forwardPE']
        # If PE ratio is under a certain threshold, it is cheap
        if price_to_book_ratio<15:
            points += .5
        elif price_to_book_ratio>18:
            points -= .5


        # Remove all data points where there is a NA entry
        data.dropna(inplace=True)

        # Update ticker points (/10 because loop runs 10 times and average is taken)
        tickers_points[ticker] += float(points/10)

        # Plot the graph (debugging)
        #data[['Close', 'SMA30', 'EWMA30', 'Upper Band', 'Lower Band']].plot(label=ticker,figsize=(16, 8))
        #plt.show()

        # Show all data (debugging)
        #print(data)

# Print out relevant tickers and their score, the higher it is, the better of a buy it is
for i in sorted(tickers_points.items(), key=lambda item: item[1], reverse=True):
    print(i)

# Print out every ticker's rmse averaged over 10 loops
print(tickers_rmse)