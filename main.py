import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Fetchs stock data
ticker = 'AMZN'  # Example: Apple Inc.
start_date = '2022-01-01'
end_date = '2025-01-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Preprocess the data
data = data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences for  LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # Number of days to look back
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Predict the next day's stock price
last_sequence = scaled_data[-seq_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)
predicted_price = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"Predicted next day's closing price: ${predicted_price[0][0]:.2f}")

# Plot the results
train = data[:train_size + seq_length]
valid = data[train_size + seq_length:]
valid['Predictions'] = scaler.inverse_transform(model.predict(X_test))

plt.figure(figsize=(14, 7))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
