import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def generate_data_ffnn(array: np.ndarray, num_lags: int, num_countries: int) \
    -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates inputs and targets data for FFNN, each sample is flattened 
    
    Args:
        array: Dataset structured in the shape of (countries * timesteps, features). 
        The values must be sorted per country first and per timestep second.
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
    Generates inputs and targets data for RNN.

    Args:
        array: Dataset structured in the shape of (countries * timesteps, features). 
        The values must be sorted per country first and per timestep second.
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
    # dropping the last window of observations from each country (since there's no target for it) 
    # and flattening only the countries
    windows = windows[:, :-1, :, :]
    inputs = windows.reshape(num_countries * windows.shape[1], num_lags, num_features)
    return(inputs, targets)


def scaler(array: np.ndarray, return_scaling_parameters: bool = True, use_mean_std_list: list[float] | None = None) -> tuple:
    '''
    Normalises the input variables

    Args:
        array: numpy array of variables, of shape (num_of_samples, num_of_variables)
        return_scaling_parameters: specifies whether to include the scaling parameters 
        for the last column variable in the tuple. Default is True
        use_mean: optional, list with pre-calculated mean and standard deviation that is to be used when scaling the array

    Returns:
        Tuple: tuple with normalised array. If return_scaling_parameters is set to True, 
        then returns (normalised_array, mean, std), where the scaling metrics are arrays for each of the columns
    '''
    mean = array.mean(axis = 0)
    std = array.std(axis = 0)
    if use_mean_std_list == None:
        scaled_arr = (array - mean) / std
    else:
        scaled_arr = (array - use_mean_std_list[0]) / use_mean_std_list[1]
    if return_scaling_parameters:
        return(scaled_arr, mean, std)
    else:
        return scaled_arr

def differencer(array: np.ndarray, num_countries: int, column_slices_to_difference: list[int]) -> np.ndarray:
    '''
        Produces an array of first differences along rows for long-format panel data, meaning each country is differenced independently. NaNs are removed 
    
        Args:
            array: numpy array of variables, of shape (num_of_samples, num_of_variables)
            num_countries: number of countries in the training data
            column slices to difference: list containing indexes of columns of variables that need to be differenced in a given array
           
        Returns:
            array: long-format panel array of differences
        '''
    num_timesteps = array.shape[0] // num_countries
    num_features = array.shape[1]
    # reshaping into (num_countries, num_timesteps, num_features), where each matrix is one country
    reshaped_arr = array.reshape(num_countries, num_timesteps, -1).copy()
    # differencing only the given variables' columns while leaving the first element in this columns intact so that the shapes match
    reshaped_arr[:, 1:, column_slices_to_difference] = np.diff(reshaped_arr[:, :, column_slices_to_difference], axis = 1)
    # dropping the first time step in each country, as it was not differenced and reshaping
    differenced_arr = reshaped_arr[:, 1:, :].reshape(num_countries * (num_timesteps - 1), num_features)
    return differenced_arr