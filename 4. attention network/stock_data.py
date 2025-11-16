import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pickle

start = "2000-01-01"
end = "2025-11-13"
sp500 = yf.download("^GSPC", start, end)
dow = yf.download("^DJI", start, end)
nasdaq = yf.download("^IXIC", start, end)

price = pd.DataFrame()
price["SP500"] = sp500["Close"]
price["DOW"] = dow["Close"]
price["NASDAQ"] = nasdaq["Close"]
price.dropna(inplace= True)

def trend(y):
    x = np.arange(y.size).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model.predict(x)

# find trend of stock price
for col in price.columns:
    price[f"{col}_trend"] = trend(price[col])
price.plot()
plt.title("stock prices")
plt.show()

# removing trend from stock price
for col in price.columns[: 3]:
    price[f"{col}_rt"] = price[col] - price[f"{col}_trend"]
price.iloc[: , -3:].plot()
plt.title("stock prices with trend removed")
plt.show()

# normalize stock price
price = np.array(price)[: , -3: ]
mean = np.mean(price, axis= 0)
std = np.std(price, axis= 0)
price = (price - mean) / std
plt.plot(price[: , -3: ])
plt.title("Normalized stock prices")
plt.axhline(0)
plt.show()

# split the price data into training data and test data
x_train = price[: -20]
x_test = price[-20: ]

# generate training dataset for transformer model
seq_len = 60
n = x_train.shape[0]
m = range(n - 2*seq_len + 1)
xi_enc = np.array([x_train[i: i + seq_len] for i in m])
xi_dec = np.array([x_train[i + seq_len - 1: i + seq_len*2 - 1] for i in m])
xo_dec = np.array([x_train[i + seq_len: i + seq_len*2] for i in m])

with  open("stock_data.pkl", "wb") as f:
    pickle.dump([x_train, x_test, xi_enc, xi_dec, xo_dec], f)







