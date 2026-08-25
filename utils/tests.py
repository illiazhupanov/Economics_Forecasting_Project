import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def check_stationarity(df):
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