import numpy as np
import os 
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from keras import ops
from utils.preprocessing import scaler, differencer, generate_data_ffnn
import matplotlib.pyplot as plt


class ffnn:
    def __init__(self, train_array: np.ndarray, val_array: np.ndarray | None = None,
                 num_countries: int = 22, num_lags: int = 3):
        '''
        Initialises the instance of a class with the data that model will be trained on

        Args:
            train_array: array of train data, of shape (samples, features), in long-format panel data 
            num_countries: number of countries in the training data
            val_array: optional, an array of validation data, of shape (samples, features), in long-format panel data 
            num_lags: number of lagged time steps that will be used to forecast the next one
        '''
        self.train_array = train_array
        self.val_array = val_array
        self.num_countries =  num_countries
        self.num_lags = num_lags
        self.model_initialised = False


    def init_model(self, optimiser: str = 'adam', dropout_rate: int = 0.2, loss: str = 'mse'):
        '''
        Initialises the instance of model used inside class

        Args:
            optimiser: optimiser used in training. Default is adam
            dropout_rate: dropout rate used in dropout layer to prevent overfitting. Default is 0.2
            loss: loss function used in training. Default is MSE
        '''
        self.model_initialised = True
        num_features = self.train_array.shape[1]
        inputs = keras.layers.Input(shape = (num_features * self.num_lags,))
        features = keras.layers.Dense(16, activation = 'relu')(inputs)
        features = keras.layers.Dropout(dropout_rate)(features)
        features = keras.layers.Dense(8, activation = 'relu')(features)
        outputs = keras.layers.Dense(1)(features)
        self.model = keras.Model(inputs = inputs, outputs = outputs)
        self.model.compile(optimizer = optimiser, loss = loss)


    def fit_model_once(self, list_of_column_slices_to_difference: list[int], difference: bool = True, 
                  normalise: bool = True, batch_size: int = 32, epochs: int = 20, 
                  tuning: bool = False):
        '''
        Fits the initialised model once. Can be run with tuning = True to be fit with validation for tuning, 
        given that a validation array was passed in at instantiation
        
        Args:
            
                '''
        if not self.model_initialised:
            self.init_model()
        train_arr = self.train_array.copy()
        if difference:
            train_arr = differencer(train_arr, self.num_countries, list_of_column_slices_to_difference)
            if tuning:
                val_arr = self.val_array.copy()
                val_arr = differencer(val_arr, self.num_countries, list_of_column_slices_to_difference)
        if normalise:
            # the mean and std returned are for the last column, which is unemployment in my case
            train_arr, mean_variable, std_variable = scaler(train_arr)
            if tuning:
                val_arr = scaler(val_arr, return_scaling_parameters = False, 
                                 use_mean_std_list = (mean_variable, std_variable))
                
        train_inputs, train_targets = generate_data_ffnn(train_arr, self.num_lags, self.num_countries)
        if not tuning:
            pass
            ################################################
        else: 
            val_inputs, val_targets = generate_data_ffnn(val_arr, self.num_lags, self.num_countries)
            history = self.model.fit(train_inputs, train_targets, batch_size = batch_size, epochs = epochs, 
                           validation_data = (val_inputs, val_targets))
            plt.plot(history.history['loss'], label = 'training loss ')
            plt.plot(history.history['val_loss'], label = 'Validation loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
        

