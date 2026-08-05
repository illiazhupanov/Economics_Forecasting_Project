import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
def generate_data_ffnn(array: np.ndarray, number_of_lags: int, num_countries: int, num_timesteps: int, num_features: int) \
    -> tuple[np.ndarray, np.ndarray]:
    '''takes the structured 2-dimensional numpy dataset as an input and a desired number of lags.
    Returns a tuple with train and targets arrays'''
    # reshaping the array to get a matrix per country
    array = array.reshape(num_countries, num_timesteps, num_features) 
    windows = sliding_window_view(x = array, window_shape = number_of_lags, axis = 1)
    # extracting the targets into a vector and flattening
    targets = windows[:, 1:, -1, -1] 
    targets = targets.reshape(-1,)
    # dropping the last window of observations from each country (since there's no target for it) and flattening
    windows = windows[:, :-1, :, :]
    inputs = windows.reshape(num_countries * windows.shape[1], number_of_lags * num_features)
    return(inputs, targets)