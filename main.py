import math
import pandas as pd
import pandas_datareader as web
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

plt.style.use('fivethirtyeight')

#pobieranie notowań giełdowych dla firmy aple
firma = 'AAPL'  #skrót firmy dla której chcemy pobrać dane w tym przypadku Apple
df = web.DataReader(firma, data_source='yahoo', start='2012-01-01', end='2022-06-10')

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
#przygotowanie danych
scaler = MinMaxScaler(feature_range=(0, 1))  #skalowanie danych
scaler_data = scaler.fit_transform(dataset)

train_data = scaler_data[0:training_data_len, :]
#przygotowanie danych do trenowania

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
#konwertowanie x i y na numpy array a następnie przekształcanie danych
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#tworzenie modelu
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')  #kompilowanie modelu

model.fit(x_train, y_train, batch_size=1, epochs=1)  #trenowanie modelu
#wczytanie zbioru danych testowych i rozpoczęcie przewidywań
test_data = scaler_data[training_data_len-60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#uzyskiwanie przewidywanych wartości dla modelu
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#wizualizacja wykresu
plt.figure(figsize=(16, 8))
plt.title(firma)
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
