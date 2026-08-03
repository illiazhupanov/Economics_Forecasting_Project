import numpy as np
import pandas as pd
def generate_data_ffnn(array: np.ndarray, number_of_lags: int) -> tuple[np.ndarray, np.ndarray]:
    '''takes the structured 2-dimensional numpy dataset as an input and a desired number of lags.
    Returns a tuple with train and targets arrays'''

    