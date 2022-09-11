import websocket
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pyfolio
import pandas_ta   





data = yf.download('AAPL', period="ytd", auto_adjust=True)
data = data['Close'].to_frame()
data['SMA30'] = data['Close'].rolling(30).mean()
data['EWMA30'] = data['Close'].ewm(span=30).mean()

# calculate Simple Moving Average with 20 days window
sma = data['Close'].rolling(window=20).mean()
# calculate the standar deviation
rstd = data['Close'].rolling(window=20).std()
data['Upper Band'] = sma + 2 * rstd
data['Lower Band'] = sma - 2 * rstd

data['RSI'] = pandas_ta.rsi(data['Close'])


data.dropna(inplace=True)

data[['Close', 'SMA30', 'EWMA30', 'Upper Band', 'Lower Band']].plot(label='AAPL',figsize=(16, 8))
#plt.show()

print(data)

