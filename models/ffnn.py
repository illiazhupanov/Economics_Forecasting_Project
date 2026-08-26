import numpy as np
import os 
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from keras import ops
from utils.preprocessing import scaler, difference, generate_data_ffnn


class ffnn:
    def __init__(self, train_array: np.ndarray, val_array: np.ndarray, num_previous_timesteps: int, num_countries: int = 22):
        '''
        Initialises the instance of a class with the data that model will be trained on

        Args:
            train_array: array of train data, of shape (samples, features), in long-format panel data 
            val_array: array of validation data, of shape (samples, features), in long-format panel data 
            num_previous_timesteps: number of previous timesteps to be used to produce the outputs
            num_countries: number of countries in the training data
        '''
        self.train_array = train_array
        self.val_array = val_array
        self.num_previous_timesteps = num_previous_timesteps
        self.num_countries =  num_countries

    def init_model(self, optimiser: str = 'adam', dropout_rate: int = 0.2, loss: str = 'mse'):
        '''
        Initialises the instance of model used inside class

        Args:
            optimiser: optimiser used in training. Default is adam
            dropout_rate: dropout rate used in dropout layer to prevent overfitting. Default is 0.2
            loss: loss function used in training. Default is MSE
        '''
        num_features = self.__train_array.shape[1]

        inputs = keras.layers.Input(shape = (num_features * self.__num_previous_timesteps))
        features = keras.layers.Dense(16, activation = 'relu')(inputs)
        features = keras.layers.Dropout(dropout_rate)(features)
        features = keras.layers.Dense(8, activation = 'relu')(features)
        outputs = keras.layers.Dense(1)(features)

        self.__model = keras.Model(inputs = inputs, outputs = outputs)
        self.__model.compile(optimizer = optimiser, loss = loss)

    def fit_model(self):
        

    



