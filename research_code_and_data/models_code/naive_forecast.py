import pandas as pd
import numpy as np
import os 
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from utils.misc import generate_timesteps_list

def naive_forecast(df, test_years: list[int]):
    '''
    Produce a naive forecast for each country (repeating the last-known value)

    Args: 
        df: a pandas dataframe with long-format panel data, where the first column is the reference area, 
        the second column is the is the time steps, and the last column the forecast variable
        test_years: list with (first_year, last_year) inclusively, data for that time period will be used 

    Returns:
        pandas dataframe with both the original values and forecasts over the test period
    '''
    df = df.copy()
    test_timesteps_list = generate_timesteps_list(*test_years)
    df['Forecasts'] = df.groupby(df.columns[0])[df.columns[-1]].shift(1)
    result_df = df.iloc[:, [0, 1, -1]]
    result_df = df.loc[df[df.columns[1]].isin(test_timesteps_list), :].reset_index(drop=True)
    return result_df
