from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import pickle
from my_utils import *

LOSS_FUNC = 'MSE'
OPTIMIZER = 'adam'

def model_for_playing(D):
    
    model = Sequential()
    model.add(Dense(D, input_dim = D, activation="sigmoid", kernel_regularizer=l2(1e-5)))
#     model.add(Dropout(0.6))

#     model.add(Dense(256, activation="relu"))
#     model.add(Dropout(0.6))

    model.add(Dense(200, activation="relu", kernel_regularizer=l2(1e-5)))
#     model.add(Dropout(0.6))
# 
    model.add(Dense(100, activation="relu", kernel_regularizer=l2(1e-5)))
#     model.add(Dropout(0.6))
# 
#     model.add(Dense(64, activation="relu"))
#     model.add(Dropout(0.6))
    model.add(Dense(1, activation="linear", kernel_regularizer=l2(1e-5)))

    model.compile(
        loss=LOSS_FUNC,
        optimizer=OPTIMIZER)
    return model

def model_for_hokming(D):
    
    model = Sequential()
    model.add(Dense(56, input_dim = D, activation="sigmoid"))
#     model.add(Dropout(0.6))

    model.add(Dense(56, activation="relu"))
#     model.add(Dropout(0.6))

#     model.add(Dense(32, activation="relu"))
#     model.add(Dropout(0.6))

    model.add(Dense(32, activation="relu"))
#     model.add(Dropout(0.6))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss=LOSS_FUNC,
        optimizer=OPTIMIZER)
    return model

class LearningModel:
    # the model for playing the 
    def __init__(self, _for = 'Playing', _type = 'SGD', warm_up = False, n_trained = 0):
        self._n_trained = n_trained
        self._for = _for
        self._type = _type
        D = DIMENSION
        if warm_up:
            self.load_model()
        else:
            if _for == 'Playing':
                if _type == 'DNN':
                    self.model = model_for_playing(D)
                elif _type == 'SGD':
                    self.model = np.random.randn(D) / np.sqrt(D)
                elif _type == 'SKLearn':
                    self.model = SGDRegressor(loss='squared_loss', penalty='l2', verbose=0, learning_rate='invscaling')
                    self.model.partial_fit(np.atleast_2d([np.random.choice([0,1]) for _ in range(D)]), [0])
            
            elif _for == 'Hokming':
                D = N_CARDS + len(CARD_TYPES)
                if _type == 'DNN':
                    self.model = model_for_hokming(D)
                elif _type == 'SGD':
                    self.model = np.random.randn(D) / np.sqrt(D)
                elif _type == 'SKLearn':
                    self.model = SGDRegressor(loss='squared_loss', penalty='l2', verbose=0, learning_rate='invscaling')
                    self.model.partial_fit(np.atleast_2d([np.random.choice([0,1]) for _ in range(D)]), [0])
    
    def partial_fit(self, X, Y, lr):
        X = np.atleast_2d(X)
        if self._type == 'DNN':
            self.model.fit(X, Y, epochs = 1, verbose = 0)
        elif self._type == 'SGD':
            self.model += lr * (Y - X.dot(self.model)/X.shape[0]).dot(X)
        elif self._type == 'SKLearn':
            self.model.partial_fit(X, Y)
        self._n_trained += 1
    
    def predict(self, X):
        if self._type == 'DNN':
            return self.model.predict(np.atleast_2d(X))[0][0]
        elif self._type == 'SGD':
            return X.dot(self.model)
        elif self._type == 'SKLearn':
            return self.model.predict(np.atleast_2d(X))[0]
    
    def save(self):
        f_name = f"./{self._for}-{self._type}"
        if self._type == 'DNN':
            self.model.save(f"./saved_models/{f_name}.h5")
        elif self._type == 'SGD':
            df = pd.DataFrame(self.model, columns = ['Coeffs'])
            df.to_csv(f"./saved_models/{f_name}.csv")
        elif self._type == 'SKLearn':
            with open(f"./saved_models/{f_name}.pkl", 'wb') as file:
                pickle.dump(self.model, file)
        return self._n_trained
    
    def load_model(self):
        f_name = f"./{self._for}-{self._type}"
        if self._type == 'DNN':
            self.model = load_model(f"./saved_models/{f_name}.h5")
            self.model.compile(loss=LOSS_FUNC, optimizer=OPTIMIZER)
        elif self._type == 'SGD':
            self.model = np.array(pd.read_csv(f"./saved_models/{f_name}.csv", index_col = 0)).reshape(1, -1)[0]
        elif self._type == 'SKLearn':
            with open(f"./saved_models/{f_name}.pkl", 'rb') as file:
                pickle.load(file)
        return self._n_trained
        