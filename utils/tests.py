import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import itertools
from scipy import stats

def check_stationarity_dataframe(df):
    '''
    Check how many differencing orders does it take to make a time series stationary and produce 
    the p-value of the Augmented Dickey-Fuller test, as well as the plot of differenced time series

    Args:
        df: a pandas dataframe of time series data, where first column is the timesteps, and the second is the variable
    '''
    diff_order = 0
    variable = df.iloc[:, 1]
    while True:
        p_value = adfuller(variable)[1]
        if p_value < 0.05:
            break
        else:
            variable = variable.diff()
            variable = variable.dropna()
            diff_order += 1
    print(f'Series is stationary with p-value of {p_value} after differencing {diff_order} time(s)')
    plt.plot(df.iloc[diff_order:, 0], variable)
    plt.xticks(df.iloc[diff_order:, 0][::30])
    plt.show()

def check_stationarity(df, list_of_countries):
     '''
        Check whether the variable time series are stationary for each country with ADF and KPSS tests
    
        Args:
            df: long-format panelpandas dataframe, containing the variable
            list_of_countries: list of strings of country names 
        '''
     for country in list_of_countries:
        p_value_adfuller = adfuller(df.loc[df['Reference area'] == country, ['OBS_VALUE']])[1]
        p_value_kpss = kpss(df.loc[df['Reference area'] == country, ['OBS_VALUE']])[1]
        if p_value_adfuller >= 0.05 and p_value_kpss <= 0.05:
            print(f'{country} is non-stationary with p-value_adfuller of {p_value_adfuller} and p-value_kpss {p_value_kpss}')


def diebold_mariano(naive_df, ffnn_df, rnn_df, arimax_df):
    '''
    Runs a Diebold-Mariano test, while averaging the mean absolute errors across countries for each timestep. 
    Significance level is 5%

    Args:
        naive_df: a pandas dataframe with long-format panel data, where first column should be the reference area, 
        the second column time steps, the second-to-last the true values, and the last the forecasted values
        ffnn_df: a pandas dataframe with long-format panel data, where first column should be the reference area, 
        the second column time steps, the second-to-last the true values, and the last the forecasted values
        rnn_df: a pandas dataframe with long-format panel data, where first column should be the reference area, 
        the second column time steps, the second-to-last the true values, and the last the forecasted values
        arimax_df: a pandas dataframe with long-format panel data, where first column should be the reference area, 
        the second column time steps, the second-to-last the true values, and the last the forecasted values
    '''
    # getting only the number of forecasts that corresponds to the smallest number of observations across dataframes
    # to ensure consistency, as for some models the first one or two forecasts are missing
    min_samples = min(ffnn_df.shape[0], rnn_df.shape[0], arimax_df.shape[0])
    naive_df = naive_df.iloc[-min_samples:, :]
    ffnn_df = ffnn_df.iloc[-min_samples:, :]
    rnn_df = rnn_df.iloc[-min_samples:, :]
    arimax_df = arimax_df.iloc[-min_samples:, :]

    # computing the absolute errors
    naive_df['Absolute errors'] = (naive_df.iloc[:, -2] - naive_df.iloc[:, -1]).abs()
    ffnn_df['Absolute errors'] = (ffnn_df.iloc[:, -2] - ffnn_df.iloc[:, -1]).abs()
    rnn_df['Absolute errors'] = (rnn_df.iloc[:, -2] - rnn_df.iloc[:, -1]).abs()
    arimax_df['Absolute errors'] = (arimax_df.iloc[:, -2] - arimax_df.iloc[:, -1]).abs()
    # computing the mean absolute error, averaging across countries for each timestep
    naive_abs_errors_mean = pd.pivot(naive_df, index = naive_df.columns[1], columns = naive_df.columns[0], 
                                        values = 'Absolute errors').mean(axis = 1)
    ffnn_abs_errors_mean = pd.pivot(ffnn_df, index = ffnn_df.columns[1], columns = ffnn_df.columns[0], 
                                    values = 'Absolute errors').mean(axis = 1)
    rnn_abs_errors_mean = pd.pivot(rnn_df, index = rnn_df.columns[1], columns = rnn_df.columns[0], 
                                   values = 'Absolute errors').mean(axis = 1)
    arimax_abs_errors_mean = pd.pivot(arimax_df, index = arimax_df.columns[1], columns = arimax_df.columns[0], 
                                      values = 'Absolute errors').mean(axis = 1)
    mean_errors_dict = {'naive': naive_abs_errors_mean, 'ffnn': ffnn_abs_errors_mean, 
                            'rnn': rnn_abs_errors_mean, 'arimax': arimax_abs_errors_mean}

    for (name_model_1, model_1_errors), (name_model_2, model_2_errors) \
        in itertools.combinations(mean_errors_dict.items(), 2):

        print(f'comparing {name_model_1} with {name_model_2}')
        num_errors = model_1_errors.shape[0]

        errors_diff = (model_1_errors - model_2_errors).to_numpy().reshape(num_errors,)
        mean_error_diff = np.mean(errors_diff)
        mean_error_std = np.std(errors_diff) / np.sqrt(num_errors)
        # standardising the value
        standardised_mean_error_diff = mean_error_diff / mean_error_std
        p_value = round(2 * (1 - stats.norm.cdf(abs(standardised_mean_error_diff))), 4)
        if standardised_mean_error_diff > 1.96 or standardised_mean_error_diff < -1.96:
            if mean_error_diff < 0:
                print(f'{name_model_1} is significantly better than {name_model_2} with p-value of {p_value}')
            else:
                print(f'{name_model_2} is significantly better than {name_model_1} with p-value of {p_value}')
        else:
            print(f'neither model was significantly better; p-value is {p_value}')


def MAE(df):
    '''
    Produces the mean absolute errors for every forecast for every country.

    Args: 
        df: a pandas dataframe with long-format panel data, where first column should be the reference area, 
        the second column time steps, the second-to-last the true values, and the last the forecasted values

    Returns:
        mae: a float scalar 
    '''
    mae = float((df.iloc[:, -2] - df.iloc[:, -1]).abs().mean())
    return mae

def RMSE(df):
    '''
    Produces the root mean squared errors for every forecast for every country.
    
    Args: 
        df: a pandas dataframe with long-format panel data, where first column should be the reference area, 
        the second column time steps, the second-to-last the true values, and the last the forecasted values
    
    Returns:
        rmse: a float scalar 
        '''
    rmse = np.sqrt(np.mean((df.iloc[:, -2] - df.iloc[:, -1]) ** 2))
    return float(rmse)

