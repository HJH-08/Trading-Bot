<img src="https://github.com/HJH-08/Trading-Bot/blob/main/Trading%20Bot%20Banner.png" width='1200' height = '200'>
<br>

<p align="center">
    <img src="https://img.shields.io/github/last-commit/hjh-08/Trading-Bot" />
    <img src="https://img.shields.io/github/repo-size/hjh-08/Trading-Bot">
<p>


<p align="center">
  <a href="#trading-bot">About</a> •
  <a href="#prerequisites">Prerequisites</a> •
  <a href="#potential-bugs">Potential Problems</a> 
</p>

# Trading Bot

## Using LSTM and technical indicators to predict stock price movement

This project is written in `Python`. The bot gets information of the S&P companies like historical data, current prices, volume, and data required for technical analysis from **yahoo finance**. Using `pandas dataframes` and `numpy arrays`, the data retrieved is organised, split and reshaped into training and testing data. A **LSTM (long short-term memory) model** is then created, and LSTM and dense layers with respective number of neurons are added. After training the model with historical data, the model is tested using the testing data. A root-mean squared error is calculated to evaluate the accuracy of the model. The model then predicts the price change of the stock for the next trading day. A score for the stock is calculated through evaluating and carefully weighting technical indicators (RSI, SMA ...), fundamental elements (PB, PE ratio) and the price change calculated by the machine learning model, weighted with its respective RMSE. The process is repeated 10 times and an average is taken for a more accurate prediction. Each stock's score and RMSE is printed out. Generally, a higher score indicates a better buy. (not financial advice)<br>  
The project uses:

* `pandas` databases for data manipulation
* `yfinance` to retrieve stock data
* `Matplotlib` graphs to visualise stock movement and prediction accuracy
* `pandas-ta` for technical indicators
* `numpy` arrays for high level mathematical functions
* `scikit-learn` for preparing data to train the machine learning model
* `keras` for the artificial neural network model that predicts stock movement

___
<br>
     
## Prerequisites
       
`Python` should be installed locally. Check [here](https://www.python.org/downloads/) to install depending on your OS.

### Required Modules
- `pandas`
- `yfinance`
- `matplotlib`
- `pandas-ta`
- `pandas_datareader`
- `numpy`
- `sklearn`
- `keras`


To install `pandas`:
```
$ pip install pandas
```


To install `yfinance`: 
```
$ pip install yfinance
```

To install `matplotlib`:
```
$ pip install matplotlib
```

To install `pandas-ta`:
```
$ pip install pandas-ta
```

To install `pandas_datareader`:
```
$ pip install pandas-datareader
```

To install `numpy`:
```
$ pip install numpy
```

To install `sklearn`:
```
$ pip install scikit-learn
```

To install `keras`:
```
$ pip install keras
```

<br>

### How to run the script
``` bash
$ python app.py
```
When the above code is run, something like this should appear in the terminal:
<br>

![Terminal when code is run](https://github.com/HJH-08/Trading-Bot/blob/main/Trading%20bot%20output.png)
<br>

___

## Potential Bugs

The bot might take a while to run, given the sheer volume of information that has to be retrieved for the 500 stocks, and the time taken to train the model repeatedly. ㋡ 

<br>

I am still in the process of refining this trading bot. Again, no matter how many indicators I add or how accurate I am with my calculations, the predictions that this trading bot makes is never fully accurate, and hence one should not splurge live savings and depend solely on this bot. Thanks!