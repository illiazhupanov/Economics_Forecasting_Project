import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
def generate_data_ffnn(array: np.ndarray, num_lags: int, num_countries: int) \
    -> tuple[np.ndarray, np.ndarray]:
    '''
    Generate inputs and targets data for FFNN, each sample is flattened 
    
    Args:
        array: Dataset structured in the shape of (countries * timesteps, features). The values must be sorted per country first and per timestep second.\
        The last feature column should be the targets value
        number_of_lags: Desired number of lags for input data
        num_countries: Number of countries in the training data
    
    Returns:
        tuple: tuple with inputs and targets arrays
    '''
    # getting the number of timesteps per country and the number of features
    num_timesteps = array.shape[0] // num_countries
    num_features = array.shape[1]
    # reshaping the array to get a matrix per country
    array = array.reshape(num_countries, num_timesteps, num_features) 
    # generating the sliding windows and squeezing the dimensions of size 1
    windows = sliding_window_view(x = array, window_shape = (1, num_lags, num_features))
    windows = windows.squeeze()
    # extracting the targets into a vector and flattening
    targets = windows[:, 1:, -1, -1] 
    targets = targets.reshape(-1,)
    # dropping the last window of observations from each country (since there's no target for it) and flattening both the countries and samples
    windows = windows[:, :-1, :, :]
    inputs = windows.reshape(num_countries * windows.shape[1], num_lags * num_features)
    return(inputs, targets)


def generate_data_rnn(array: np.ndarray, num_lags: int, num_countries: int) \
    -> tuple[np.ndarray, np.ndarray]:
    '''
    Generate inputs and targets data for RNN.

    Args:
        array: Dataset structured in the shape of (countries * timesteps, features). The values must be sorted per country first and per timestep second.\
        The last feature column should be the targets value
        number_of_lags: Desired number of lags for input data
        num_countries: Number of countries in the training data

    Returns:
        tuple: tuple with inputs and targets arrays
    '''
    # getting the number of timesteps per country and the number of features
    num_timesteps = array.shape[0] // num_countries
    num_features = array.shape[1]
    # reshaping the array to get a matrix per country
    array = array.reshape(num_countries, num_timesteps, num_features) 
    # generating the sliding windows
    windows = sliding_window_view(x = array, window_shape = (1, num_lags, num_features))
    windows = windows.squeeze()
    # extracting the targets into a vector and flattening
    targets = windows[:, 1:, -1, -1] 
    targets = targets.reshape(-1,)
    # dropping the last window of observations from each country (since there's no target for it) and flattening only the countries
    windows = windows[:, :-1, :, :]
    inputs = windows.reshape(num_countries * windows.shape[1], num_lags, num_features)
    return(inputs, targets)