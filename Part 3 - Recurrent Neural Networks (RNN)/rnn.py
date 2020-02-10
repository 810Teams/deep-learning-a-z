# Recurrent Neural Network


# Part 0 - Global values

def convert(data):
    data = data.tolist()
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            if isinstance(data[i][j], str):
                data[i][j] = float(data[i][j].replace(',', ''))
                
    return np.array(data)

TIMESTEPS = 60
PREDICT_ALL_AT_ONCE = False


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:].values

# Type conversion
training_set = convert(training_set)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = scaler.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = list()
y_train = list()

for i in range(TIMESTEPS, len(dataset_train)):
    X_train.append(training_set_scaled[i - TIMESTEPS:i, :])
    y_train.append(training_set_scaled[i, :])
del i
    
X_train, y_train = np.array(X_train), np.array(y_train)


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fifth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 5))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

from keras.models import load_model

try:
    # Load model
    regressor = load_model('models/model_5_inputs.h5')
except OSError:
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    
    # Save model
    regressor.save('models/model_5_inputs.h5')


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:].values

# Type conversion
real_stock_price = convert(real_stock_price)

# Getting the predicted stock price of 2017
if not PREDICT_ALL_AT_ONCE:
    dataset_total = pd.concat((dataset_train.iloc[:, 1:], dataset_test.iloc[:, 1:]), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = convert(inputs)
    inputs = scaler.transform(inputs)
    
    X_test = list()
    
    for i in range(TIMESTEPS, 80):
        X_test.append(inputs[i - TIMESTEPS:i, :]) 
    del dataset_total, inputs, i
        
    X_test = np.array(X_test)
    
    predicted_stock_price = regressor.predict(X_test)
else:
    dataset_total = dataset_train.iloc[:, 1:]
    inputs = dataset_total[len(dataset_total) - 60:].values
    inputs = convert(inputs)
    inputs = scaler.transform(inputs)
    
    predicted_stock_price = list()
    
    for i in range(TIMESTEPS, 80):
        predicted_stock_price.append(
            regressor.predict(inputs.reshape(1, inputs.shape[0], inputs.shape[1]))
        )
        inputs = np.concatenate((inputs, predicted_stock_price[-1]), axis=0)[1:,:]
    del dataset_total, inputs, i
    
    predicted_stock_price = np.array(predicted_stock_price)
    predicted_stock_price = predicted_stock_price.reshape(
        predicted_stock_price.shape[0],
        predicted_stock_price.shape[2]
    )
    
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)  

plt.plot(real_stock_price[:, 0], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price[:, 0], color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
