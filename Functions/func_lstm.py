import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

import numpy as np

def create_X(dataset, TIME_STEPS=7, gap=7):
  dataset = np.array(dataset)
  X = np.array(
      [dataset[i:(i + TIME_STEPS)] for i in range(len(dataset) - gap)]
  )
  return X

def create_y_0(arr, gap=7):
  y = np.array([arr[i + gap] for i in range(len(arr) - gap)])
  return y

def func_model(n_features=1):
    model = Sequential([
        LSTM(300, return_sequences=True, activation='tanh',
             input_shape=[None, n_features], activity_regularizer='l2'),
        # Dropout(0.1),
        LSTM(200, return_sequences=True, activation='tanh',
             activity_regularizer='l2'),
        # Dropout(0.05),
        LSTM(100, return_sequences=False, activation='tanh',
             activity_regularizer='l2'),
        # Dropout(0.05),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def func_model_4layers(n_features=1):
    model = Sequential([
        LSTM(400, return_sequences=True, activation='tanh',
             input_shape=[None, n_features], activity_regularizer='l2'),
        # Dropout(0.1),
        LSTM(300, return_sequences=True, activation='tanh',
             activity_regularizer='l2'),
        LSTM(200, return_sequences=True, activation='tanh',
             activity_regularizer='l2'),
        # Dropout(0.05),
        LSTM(100, return_sequences=False, activation='tanh',
             activity_regularizer='l2'),
        # Dropout(0.05),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def func_lstm_forecast(model, scaler_1, yy_test, TIME_STEPS=7, n_forecast=28):
  X_forecast = yy_test[-TIME_STEPS:]

  X_forecast = X_forecast.reshape((len(X_forecast), 1))[np.newaxis]

  y_forecast = []
  for i in range(n_forecast):
      y_forecast_tem = model.predict(X_forecast)
  #     print(y_forecast_tem)
      y_forecast.append(y_forecast_tem[0][0])

      X_forecast_new = np.append(X_forecast, y_forecast_tem)
      X_forecast_new = X_forecast_new[1:]
      X_forecast_new = X_forecast_new.reshape(X_forecast.shape)

      X_forecast = X_forecast_new

  y_forecast = np.array(y_forecast).reshape(-1, 1)
  y_forecast_tr = scaler_1.inverse_transform(y_forecast)
  return y_forecast_tr.reshape(-1)
