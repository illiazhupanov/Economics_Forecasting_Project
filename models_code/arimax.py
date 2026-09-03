from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np 
import os 
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from utils.misc import generate_timesteps_list
from utils.preprocessing import differencer
import matplotlib.pyplot as plt
import time
import itertools
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import json

class arimax:
    def __init__(self, df, column_names_to_difference: list[str], train_years: list[int], 
                 val_years: list[int], test_years: list[int]):
        '''
        Initialises the instance of a class of the model and its train, validation, and test data. 
        The values for exogenous variables are shifted by 1

        Args:
            df: a pandas dataframe with long-format panel data, where first column should be the reference area, 
            the second column time steps, and the last column the forecast variable
            column_names_to_difference: list of string column names, values in which should be differenced
            train_years: list with (first_year, last_year) inclusively, data for that time period will be used 
            for initial training as well as rolling-origin training
            val_years: list with (first_year, last_year) inclusively, data for that time period will be used 
            for initial tuning validation as well as rolling-origin training
            test_years: list with (first_year, last_year) inclusively, data for that time period will be used 
            for producing forecast errors during rolling-origin forecasting
        '''
        df = df.copy()
        self.original_df = df.copy()
        # differencing the columns given by the argument
        df[column_names_to_difference] = df.groupby('Reference area')[column_names_to_difference].diff()
        # shifting the values in columns with exogenous variables
        for i in range(2, len(df.columns) - 1):
            df[df.columns[-i]] = df.groupby(df.columns[0])[df.columns[-i]].shift(1) 
        self.df = df.dropna().reset_index(drop=True)
        self.list_of_countries = list(df[df.columns[0]].unique())

        # dropping the first quarter, as the exogenous data for it is not available 
        self.train_timesteps_list = generate_timesteps_list(*train_years)[2:]
        self.val_timesteps_list = generate_timesteps_list(*val_years)
        self.test_timesteps_list = generate_timesteps_list(*test_years)

    def rolling_origin_tuning(self, df_curr_country, order: tuple):
        '''
        Runs rolling-origin forecasting on validation data, refitting every time step

        Args:
            df_curr_country: pandas dataframe with data of a single country, sorted by timesteps
            order: order (p, d, q) of the model

        Returns:
            mse: mean squared error of all forecasts made for that specific country
        '''
        errors_list = []
        endog = df_curr_country.iloc[:len(self.train_timesteps_list), -1].astype(float)
        exog = df_curr_country.iloc[:len(self.train_timesteps_list), 2:-1].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters found')
            # fitting the model on training data
            model = ARIMA(endog = endog, exog = exog, order = order).fit()
            if not model.mle_retvals["converged"]:
                print('fit failed')
                return np.inf

        for cnt in range(len(self.val_timesteps_list)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                warnings.filterwarnings('ignore', message='Non-invertible starting MA parameters found')
                # re-fitting the model on training data + the validation data that has already been used
                endog = df_curr_country.iloc[:len(self.train_timesteps_list) + cnt, -1].astype(float)
                exog = df_curr_country.iloc[:len(self.train_timesteps_list) + cnt, 2:-1].astype(float)
                model = ARIMA(endog = endog, exog = exog, order = order).fit()
                if not model.mle_retvals['converged']:
                    model = ARIMA(endog=endog, exog=exog, order=order).fit(method_kwargs={"method": "nm", "maxiter": 1000})
                    if not model.mle_retvals['converged']:
                        print('fit failed')
                        return np.inf
            curr_forecast_index = len(self.train_timesteps_list) + cnt
            exog_future = df_curr_country.iloc[[curr_forecast_index], 2:-1].astype(float)
            forecast = model.forecast(steps = 1, exog = exog_future).iloc[0]
            true_value = df_curr_country.iloc[curr_forecast_index, -1]
            errors_list.append((true_value - forecast) ** 2)
        mse = np.mean(errors_list)
        return mse



    def tune_model(self):
        '''
        Tunes the model by choosing its order. Auto ARIMA is run first using AIC for a reasonable guess of the order, 
        and then grid search is performed 2 p or q around the values from auto ARIMA, while miniming MSE. 
        Best models' orders per country are saved along with their MSE
        '''
        # the best configs and their MSEs will be saved here
        print('tuning the model on validation data...')
        best_orders_dict = {country : [] for country in self.list_of_countries}

        for country in self.list_of_countries:
            df_curr_country = self.df.loc[self.df[self.df.columns[0]] == country, :]

            # running the auto arima 
            endog = df_curr_country.iloc[:len(self.train_timesteps_list), -1].astype(float)
            exog = df_curr_country.iloc[:len(self.train_timesteps_list), 2:-1].astype(float)

            auto_model = auto_arima(y = endog, X = exog, seasonal = False, suppress_warnings = True)
            auto_ar, auto_diff, auto_ma = auto_model.order
            # selecting the ranges around the p, d, q chosen by auto arima
            ar_orders_list = list(range(max(auto_ar-2, 0), auto_ar + 3))
            diff_orders_list = list(range(0, 2))
            ma_orders_list = list(range(max(auto_ma-2, 0), auto_ma + 3))

            orders = list(itertools.product(ar_orders_list, diff_orders_list, ma_orders_list))
            # parallel processing for rolling-origin tuning per-country
            results = Parallel(n_jobs = 12)(delayed(self.rolling_origin_tuning)(
                df_curr_country, order) for order in orders)
            # recording the order that gave the best MSE
            best_mse_index = np.argmin(results)
            best_order = orders[best_mse_index]
            best_mse = results[best_mse_index]
            best_orders_dict[country].append(best_order)
            best_orders_dict[country].append(best_mse)
        # saving the best configs and their MSEs
        metadata = best_orders_dict
        with open("..\\models_saves\\arimax\\arimax_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)



    def rolling_origin_forecasting(self):
        '''
        Performs rolling-origin forecasting, refitting the model at every origin. 
        If models were not tuned per country, then tune_model() is first called. If the model doesn't converge with 
        gradient-based optimiser, then Nelder-Mead gradient-free optimiser is run. For each origin, the model is re-fitted
        by warm-starting from previous origin's parameters

        Returns: 
            pandas dataframe with both the original values and forecasts over the test period
        '''
        try:
            with open('..\\models_saves\\arimax\\arimax_metadata.json', 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            self.tune_model()
            with open('..\\models_saves\\arimax\\arimax_metadata.json', 'r') as f:
                metadata = json.load(f)

        print('starting rolling-origin forecast...')
        forecasts_dict = {country: [] for country in self.list_of_countries}

        for country in self.list_of_countries:
            # resetting the parameters at each country
            prev_parameters = None
            order = metadata[country][0]
            df_curr_country = self.df.loc[self.df[self.df.columns[0]] == country, :]
            for increment in range(len(self.test_timesteps_list)):
                # getting the index of the row that contains the value that is currently being forecast (used in iloc)
                curr_forecast_index = len(self.train_timesteps_list) + len(self.val_timesteps_list) + increment
                # getting the endogenous and exogenous variables to fit the model
                endog = df_curr_country.iloc[:curr_forecast_index, -1].astype(float)
                exog = df_curr_country.iloc[:curr_forecast_index, 2:-1].astype(float)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    # warm-starting the model
                    if prev_parameters is not None:
                        model = ARIMA(endog=endog, exog=exog, order=order).fit(start_params = prev_parameters)
                    else:
                        model = ARIMA(endog=endog, exog=exog, order=order).fit()
                    # handing the case if the model did not converge with gradient-based optimisation
                    if not model.mle_retvals['converged']:
                        print(f'{country} step {increment}: fit did not converge, order={order}')
                        # re-attempting convergence with gradient-free optimisation
                        model = ARIMA(endog=endog, exog=exog, order=order).fit(method_kwargs={"method": "nm", "maxiter": 1000})
                        if not model.mle_retvals['converged']:
                            print(f'{country} step {increment}: still not converged with gradient-free optimisation')
                # updating the model parameters if it converged
                if model.mle_retvals['converged']:
                    prev_parameters = model.params
                # making a forecast
                exog_future = df_curr_country.iloc[[curr_forecast_index], 2:-1].astype(float)
                forecast = model.forecast(steps = 1, exog = exog_future).iloc[0]
                # saving the forecast
                forecasts_dict[country].append(forecast)
        result_df = self.construct_result_df(forecasts_dict)
        return result_df

    def construct_result_df(self, forecasts_dict):
        # getting only the columns of reference areas, timesteps, and the true values of forecast variable  
        # for the test period
        result_df = self.original_df.loc[self.original_df.iloc[:, 1].isin(self.test_timesteps_list), 
                                         self.original_df.columns[[0, 1, -1]]]
        combined_forecasts = [forecast for forecast_list in forecasts_dict.values() for forecast in forecast_list]
        result_df['Forecasts'] = combined_forecasts
        return result_df





