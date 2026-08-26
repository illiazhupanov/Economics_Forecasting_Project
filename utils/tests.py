import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

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
