import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#in terminal:
#conda activate tf
#d:

"""LSTM stock price "predictor" (time series prediction, regression problem; trained using backpropagation through time)"""

"""import dataset"""
FILENAME = 'data.csv'
data = pd.read_csv(FILENAME, usecols = [1])                             #return pandas.core.frame.DataFrame (with only column #1)
data_raw = data.values.astype("float32")                                #cast type 'float32'
scaler = MinMaxScaler(feature_range = (0, 1))                           #rescale the data to the range [0;1]
dataset = scaler.fit_transform(data_raw)                                #return numpy.ndarray

def getTrainTest(data, TRAIN_SIZE = 0.8):
    """Split the data (np.ndarray) into 80% train and 20% test"""
    train_len = int(TRAIN_SIZE*len(data))                               #define the border for the test and validation subsets
    train, test = data[0:train_len], data[train_len:len(data)]
    return train, test

def getXY(data, window_size):
    """Create a function to process the data (np.ndarray) into 'window_size' day look back slices. Output: train data and response"""
    """window_size is the #previous time steps which should be used as the inputs to predict the next time period"""
    
    samples_num = len(data) - window_size                               #number of samples
    time_steps_num = 1                                                  #number of time steps
    features_num = 1                                                    #number of features
    
    X, Y = [], []                                                       
    for i in range(0, samples_num):
        X.append(data[i:(i + window_size), 0])                          #X - closing the exchange at 't' 
        Y.append(data[i + window_size, 0])                              #Y - closing the exchange at 't + window_size'
    X = np.array(X).reshape(samples_num, time_steps_num, features_num)
    Y = np.array(Y)
    return X, Y

train, test = getTrainTest(dataset)
window_size = 1
train_X, train_Y = getXY(train, window_size)
test_X, test_Y = getXY(test, window_size)

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

assert(train_X.shape == (1006, 1, 1))
assert(train_Y.shape == (1006,))
assert(test_X.shape == (251, 1, 1))
assert(test_Y.shape == (251,))

def getModel(train_X, train_Y, window_size):
    """Build model, compile it and fit it here"""
    memory_units_num = 5                                                #the number of memory units, or blocks
    
    model = Sequential()
    model.add(LSTM(memory_units_num, input_shape=(1, window_size)))     #activation='tanh'(to overcome the vanishing gradient problem), recurrent_activation='sigmoid' (to forget/remember the information)
    model.add(Dense(1))                                                 #make a single value prediction
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_Y, epochs = 50, batch_size = 1, verbose = True)
    return model

model = getModel(train_X, train_Y, window_size)

def predict_and_score(model, X, Y, scaler):
    pred = scaler.inverse_transform(model.predict(X))
    orig_data = scaler.inverse_transform([Y])
    score = np.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return score

rmse_score = predict_and_score(model, test_X, test_Y, scaler)
print(rmse_score)
assert rmse_score < 10
print('You have passed the test case')
