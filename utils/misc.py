import numpy as np
import pandas as pd
import os 
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

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


def exog_coeffs_summary_CSV(exog_regressors_dict, filepath_cpi, filepath_rgdp):
    with open(filepath_cpi, 'w') as f:
        f.write(f'Country, Positive coefficients, Negative coefficients, Significant (5%)' + '\n')
    with open(filepath_rgdp, 'w') as f:
        f.write(f'Country, Positive coefficients, Negative coefficients, Significant (5%)' + '\n')
    for key, value in exog_regressors_dict.items():
        cpi_positive_coeffs = 0
        cpi_coeffs = value['CPI inflation'][::2]
        cpi_pval = value['CPI inflation'][1::2]
        for coeff in cpi_coeffs:
            if coeff > 0:
                cpi_positive_coeffs += 1
        cpi_negative_coeffs = len(cpi_coeffs) - cpi_positive_coeffs
        cpi_significant_pval = 0
        for p_value in cpi_pval:
            if p_value <= 0.05:
                cpi_significant_pval += 1
        with open(filepath_cpi, 'a') as f:
            f.write(f'{key}, {cpi_positive_coeffs}, {cpi_negative_coeffs}, {cpi_significant_pval}' + '\n')

        rgdp_positive_coeffs = 0
        rgdp_coeffs = value['RGDP growth'][::2]
        rgdp_pval = value['RGDP growth'][1::2]
        for coeff in rgdp_coeffs:
            if coeff > 0:
                rgdp_positive_coeffs += 1
        rgdp_negative_coeffs = len(rgdp_coeffs) - rgdp_positive_coeffs
        rgdp_significant_pval = 0
        for p_value in rgdp_pval:
            if p_value <= 0.05:
                rgdp_significant_pval += 1
        with open(filepath_rgdp, 'a') as f:
            f.write(f'{key}, {rgdp_positive_coeffs}, {rgdp_negative_coeffs}, {rgdp_significant_pval}' + '\n')

