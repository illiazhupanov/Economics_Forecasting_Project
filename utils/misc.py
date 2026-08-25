import numpy as np
import pandas as pd

def generate_timesteps_list(first_year: int, last_year: int) -> list:
    '''
    Generates a list of string quarterly timesteps. Can be used for selecting only a specific time period from a dataframe

    Args:
        first_year: first year (inclusive) for which to generate the timesteps
        last_year: last year (inclusive) for which to generate the timesteps
    
    Returns:
        list of string timesteps in the YYYY-Q№ format
    '''
    range_years = last_year - first_year + 1
    timestep_list = []
    for curr_increment in range(range_years):
        curr_year = str(first_year + curr_increment)
        for curr_quarter in range(1, 5):
            curr_timestep = f'{curr_year}-Q{str(curr_quarter)}'
            timestep_list.append(curr_timestep)
    return timestep_list