# Cryptocurrency-Price-Prediction-using-RNN
Recurrent Neural Networks (RNNs) are used for real-time cryptocurrency price prediction, aiming to outperform traditional methods in the  volatile crypto market. Here, three RNN methods are compared  to find out which model gives the best result.


CODE FOR THIS PROJECT:


# Install required libraries
```
!pip install yfinance numpy pandas matplotlib scikit-learn tensorflow

import yfinance as yf
import pandas as pd
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
```

# Set seeds for reproducibility
```
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

# Define the tickers for Bitcoin, Ethereum, and Litecoin
```
cryptos = ['BTC-USD', 'ETH-USD', 'LTC-USD']
```

# Define the start and end date for the data
```
start_date = '2019-01-01'
end_date = '2024-01-01'
```

# Create an empty dictionary to store the data
```
crypto_data = {}
```

# Download and store data in the dictionary
```
for ticker in cryptos:
    crypto_data[ticker] = yf.download(ticker, start=start_date, end=end_date)

    # Display the first few rows of each dataframe
    print(f"Showing data for {ticker}:")
    display(crypto_data[ticker].head())

from sklearn.preprocessing import MinMaxScaler
```

# Assuming we're working with Bitcoin data for this example
```
btc_data = crypto_data['BTC-USD']
```

# 1. Handle missing data
```
# For simplicity, we'll fill missing values with the last available value
btc_data.fillna(method='ffill', inplace=True)
```

# 2. Normalize the features
```
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_btc = scaler.fit_transform(btc_data[['Open', 'High', 'Low', 'Close', 'Adj Close']])
```

# 3. Convert the data into a time series format
```
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, -1])  # Assuming close price is the target variable
    return np.array(X), np.array(Y)
```
# Define look_back period
```
look_back = 60  # for example, using past 60 days to predict the next day
```

# Split into features (X) and target (Y)
```
X_btc, Y_btc = create_dataset(scaled_data_btc, look_back)
```

# Reshape features into the format expected by LSTM (samples, time steps, features)
```
X_btc = np.reshape(X_btc, (X_btc.shape[0], look_back, X_btc.shape[2]))
```

# Split into train and test sets
```
split_percent = 0.80
split = int(split_percent * len(X_btc))

X_train_btc = X_btc[:split]
Y_train_btc = Y_btc[:split]
X_test_btc = X_btc[split:]
Y_test_btc = Y_btc[split:]

print("Data preprocessed and ready for model training.")
```

# Import TensorFlow for model training
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Bidirectional
```

# Define the LSTM model
```
lstm_model = Sequential()
lstm_model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_btc.shape[1], X_train_btc.shape[2])))
lstm_model.add(LSTM(units=100))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
```

# Define the GRU model
```
gru_model = Sequential()
gru_model.add(GRU(units=100, return_sequences=True, input_shape=(X_train_btc.shape[1], X_train_btc.shape[2])))
gru_model.add(GRU(units=100))
gru_model.add(Dense(1))

gru_model.compile(optimizer='adam', loss='mean_squared_error')
```

# Define the BiLSTM model
```
bilstm_model = Sequential()
bilstm_model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(X_train_btc.shape[1], X_train_btc.shape[2])))
bilstm_model.add(Bidirectional(LSTM(units=100)))
bilstm_model.add(Dense(1))
bilstm_model.compile(optimizer='adam', loss='mean_squared_error')
```

# Train the models
```
lstm_history = lstm_model.fit(X_train_btc, Y_train_btc, epochs=100, batch_size=32, validation_data=(X_test_btc, Y_test_btc), verbose=1)
gru_history = gru_model.fit(X_train_btc, Y_train_btc, epochs=100, batch_size=32, validation_data=(X_test_btc, Y_test_btc), verbose=1)
bilstm_history = bilstm_model.fit(X_train_btc, Y_train_btc, epochs=100, batch_size=32, validation_data=(X_test_btc, Y_test_btc), verbose=1)
```

# Evaluate the models
```
lstm_scores = lstm_model.evaluate(X_test_btc, Y_test_btc, verbose=0)
gru_scores = gru_model.evaluate(X_test_btc, Y_test_btc, verbose=0)
bilstm_scores = bilstm_model.evaluate(X_test_btc, Y_test_btc, verbose=0)

print('LSTM Test loss:', lstm_scores)
print('GRU Test loss:', gru_scores)
print('BiLSTM Test loss:', bilstm_scores)
```

# Plot training and validation loss
```
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.plot(lstm_history.history['loss'], label='LSTM training Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.title('LSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 3, 2)
plt.plot(gru_history.history['loss'], label='GRU training Loss')
plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')
plt.title('GRU Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()



plt.subplot(1, 3, 3)
plt.plot(bilstm_history.history['loss'], label='Bi-LSTM training Loss')
plt.plot(bilstm_history.history['val_loss'], label='Bi-LSTM Validation Loss')
plt.title('Bi-LSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
```

# Save the plot to the current directory
```
plt.savefig('btc-loss.png')

plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
```

# Make predictions
```
lstm_predictions = lstm_model.predict(X_test_btc)
gru_predictions = gru_model.predict(X_test_btc)
bilstm_predictions = bilstm_model.predict(X_test_btc)
```

# Calculate MSE and MAE for LSTM
```
lstm_mse = mean_squared_error(Y_test_btc, lstm_predictions)
lstm_mae = mean_absolute_error(Y_test_btc, lstm_predictions)
```


# Calculate MSE and MAE for GRU
```
gru_mse = mean_squared_error(Y_test_btc, gru_predictions)
gru_mae = mean_absolute_error(Y_test_btc, gru_predictions)
```

# Calculate MSE and MAE for BiLSTM
```
bilstm_mse = mean_squared_error(Y_test_btc, bilstm_predictions)
bilstm_mae = mean_absolute_error(Y_test_btc, bilstm_predictions)
```

# Print the scores
```
print(f'LSTM MSE: {lstm_mse}, MAE: {lstm_mae}')
print(f'GRU MSE: {gru_mse}, MAE: {gru_mae}')
print(f'BiLSTM MSE: {bilstm_mse}, MAE: {bilstm_mae}')
```

# Generate date range for x-axis
```
date_range = pd.date_range(start=start_date, end=end_date, periods=len(Y_test_btc))
```

# Plot actual vs. predicted prices for the LSTM model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_btc), scaled_data_btc.shape[1]-1)), Y_test_btc.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='b')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(lstm_predictions), scaled_data_btc.shape[1]-1)), lstm_predictions), axis=1))[:, -1], label='Predicted Prices', color='red')
plt.title('For BTC Comparison of Actual and LSTM Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)
```

# Save the plot to the current directory
```
plt.savefig('btc-lstm.png')
plt.show()
```

# Plot actual vs. predicted prices for the GRU model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_btc), scaled_data_btc.shape[1]-1)), Y_test_btc.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='b')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(gru_predictions), scaled_data_btc.shape[1]-1)), gru_predictions), axis=1))[:, -1], label='Predicted Prices', color='red')
plt.title('For BTC Comparison of Actual and GRU Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)
```

# Save the plot to the current directory
```
plt.savefig('btc-gru.png')
plt.show()
```

# Plot actual vs. predicted prices for the Bi-LSTM model
```
plt.figure(figsize=(10, 6))
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(Y_test_btc), scaled_data_btc.shape[1]-1)), Y_test_btc.reshape(-1, 1)), axis=1))[:, -1], label='Actual Prices', color='b')
plt.plot(date_range, scaler.inverse_transform(np.concatenate((np.zeros((len(bilstm_predictions), scaled_data_btc.shape[1]-1)), bilstm_predictions), axis=1))[:, -1], label='Predicted Prices', color='red')
plt.title('For BTC Comparison of Actual and Bi-LSTM Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to each year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format date to only show years
plt.legend()
plt.grid(True)
```

# Save the plot to the current directory
```
plt.savefig('btc-bilstm.png')
plt.show()

def calculate_rmse(actuals, predictions):
    """
    Calculate Root Mean Squared Error
    """
    mse = np.mean((actuals - predictions) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mape(actuals, predictions):
    """
    Calculate Mean Absolute Percentage Error
    """
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mape



# Assuming Y_test, lstm_predictions, gru_predictions, and bilstm_predictions are already defined
```

# Calculate RMSE for each model
```
lstm_rmse = calculate_rmse(Y_test_btc, lstm_predictions.flatten())
gru_rmse = calculate_rmse(Y_test_btc, gru_predictions.flatten())
bilstm_rmse = calculate_rmse(Y_test_btc, bilstm_predictions.flatten())
```

# Calculate MAPE for each model
```
lstm_mape = calculate_mape(Y_test_btc, lstm_predictions.flatten())
gru_mape = calculate_mape(Y_test_btc, gru_predictions.flatten())
bilstm_mape = calculate_mape(Y_test_btc, bilstm_predictions.flatten())
```

# Print RMSE and MAPE for each model
```
print(f'LSTM RMSE: {lstm_rmse:.3f}, MAPE: {lstm_mape:.2f}%')
print(f'GRU RMSE: {gru_rmse:.3f}, MAPE: {gru_mape:.2f}%')
print(f'BiLSTM RMSE: {bilstm_rmse:.3f}, MAPE: {bilstm_mape:.2f}%')
```


# do the same for Ethereum and Litecoin.





