import numpy as np
import pandas as pd
import os 
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from keras import ops
from utils.preprocessing import scaler, generate_data_rnn
from utils.misc import generate_timesteps_list
import matplotlib.pyplot as plt
import itertools
import time

########################## not started
class rnn:
    def __init__(self, df: pd.DataFrame, column_names_to_difference: list[str], train_years: list[int], 
                 val_years: list[int], test_years: list[int], num_countries: int = 22, num_lags: int = 3):
        '''
        Initialises the instance of a class with the data that model will be trained on

        Args:
            df: long-format panel dataframe with time series of variables 
            where columns should be ordered as follows: country name, timestep in the format of YYYY-Q№, variables,
            and the last column is the variable that is being forecast
            train_years: list with (first_year, last_year) inclusively, data for that time period will be used 
            for initial training as well as rolling-origin training
            val_years: list with (first_year, last_year) inclusively, data for that time period will be used 
            for initial tuning validation as well as rolling-origin training
            test_years: list with (first_year, last_year) inclusively, data for that time period will be used 
            for producing forecast errors during rolling-origin forecasting
            column_names_to_difference: list of column names of variables that need to be differenced in the original dataframe
            num_countries: number of countries in the training data
            num_lags: number of lagged time steps that will be used to forecast the next one
        '''
        self.original_df = df.copy()
        self.list_of_countries = list(df['Reference area'].unique())
        self.num_countries =  num_countries
        self.num_lags = num_lags
        self.model = None
        self.epochs = None
        self.batch_size = None

        # saving the undifferenced array
        ro_train_timesteps_list = [train_years[0], val_years[1]]
        train_timesteps_list = generate_timesteps_list(*ro_train_timesteps_list)
        original_ro_train_df = df.loc[df['TIME_PERIOD'].isin(train_timesteps_list), :]
        # dropping the country name and timestep indexes
        self.original_ro_train_array = original_ro_train_df.to_numpy()[:, 2:].astype(np.float32)
        
        test_timesteps_list = generate_timesteps_list(*test_years)
        self.test_timesteps_list = test_timesteps_list
        original_test_df = df.loc[df['TIME_PERIOD'].isin(test_timesteps_list), :]
        self.original_test_array = original_test_df.to_numpy()[:, 2:].astype(np.float32)

        # differencing and saving the arrays
        df[column_names_to_difference] = df.groupby('Reference area')[column_names_to_difference].diff()
        df = df.dropna()

        train_timesteps_list = generate_timesteps_list(*train_years)
        train_df = df.loc[df['TIME_PERIOD'].isin(train_timesteps_list), :]
        # dropping the country name and timestep indexes
        self.train_array = train_df.to_numpy()[:, 2:].astype(np.float32)

        val_timesteps_list = generate_timesteps_list(*val_years)
        val_df = df.loc[df['TIME_PERIOD'].isin(val_timesteps_list), :]
        self.val_array = val_df.to_numpy()[:, 2:].astype(np.float32)

        test_df = df.loc[df['TIME_PERIOD'].isin(test_timesteps_list), :]
        self.test_array = test_df.to_numpy()[:, 2:].astype(np.float32)


    def init_model(self, hidden_first_neurons, hidden_second_neurons, dropout_rate, 
                   optimiser: str = 'adam', loss: str = 'mse', ):
        '''
        Initialises the instance of model
        
        Args:
            hidden_first_neurons: the number of neurons to be used at first hidden layer
            hidden_second_neurons: the number of neurons to be used at second hidden layer
            dropout_rate: the dropout rate to be used at dropout layer
            optimiser: optimiser used in training. Default is adam
            loss: loss function used in training. Default is MSE
        
        Returns:
            A compiled instance of keras.Model with given parameters
        '''
        num_features = self.train_array.shape[1]
        inputs = keras.layers.Input(shape = (num_features * self.num_lags,))
        features = keras.layers.Dense(hidden_first_neurons, activation = 'relu')(inputs)
        features = keras.layers.Dropout(dropout_rate)(features)
        features = keras.layers.Dense(hidden_second_neurons, activation = 'relu')(features)
        outputs = keras.layers.Dense(1)(features)
        model = keras.Model(inputs = inputs, outputs = outputs)
        model.compile(optimizer = optimiser, loss = loss)
        return model


    def tune_model(self, epochs: int = 100):
        '''
        Tunes the model across a set of parameters from grid search. The best model config is saved 

        Args:
            epochs: the maximum number of epochs each model configuration can get fitted for. Default is 200
        '''
        train_arr = self.train_array.copy()
        val_arr = self.val_array.copy()
        # the mean and std returned are for the last column, which is unemployment in my case
        train_arr, mean_variable, std_variable = scaler(train_arr)
        val_arr = scaler(val_arr, return_scaling_parameters = False, 
                        use_mean_std_list = (mean_variable, std_variable))
                        
        train_inputs, train_targets = generate_data_ffnn(train_arr, self.num_lags, self.num_countries)
        val_inputs, val_targets = generate_data_ffnn(val_arr, self.num_lags, self.num_countries)

        #initialising the possible hyperparameters
        hidden_first = [16, 32]
        hidden_second = [8, 16]
        dropout_rate = [0.1, 0.2, 0.3]
        batch_sizes = [16, 32, 64]

        
        best_val_loss = float('inf')

        for h1, h2, dropout, batch_size in itertools.product(hidden_first, hidden_second, dropout_rate, batch_sizes):

            model = self.init_model(h1, h2, dropout)
            early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
            history = model.fit(train_inputs, train_targets, batch_size = batch_size, epochs = epochs, 
                       validation_data = (val_inputs, val_targets), callbacks = [early_stop], verbose = 0)
            val_loss = history.history['val_loss']
            epoch_of_best_val_loss = np.argmin(val_loss) + 1
            config_best_val_loss = np.min(val_loss)

            # saving the best config and its parameters in case it reached the val loss lower than the previous best
            if config_best_val_loss < best_val_loss:
                best_val_loss = config_best_val_loss
                best_config = (h1, h2, dropout)
                best_config_history = history.history
                self.batch_size = batch_size
                self.epochs = epoch_of_best_val_loss
                self.model = model


        print(f'best config is:\n1st hidden layer neurons: {best_config[0]}')
        print(f'2nd hidden layer neurons {best_config[1]}')
        print(f'dropout rate: {best_config[2]}')
        print(f'batch size is: {self.batch_size}')
        print(f'minimum validation loss reached: {best_val_loss}\nreached after {self.epochs} epochs')
        plt.plot(best_config_history['loss'], label = 'training loss ')
        plt.plot(best_config_history['val_loss'], label = 'validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()


    
    def unnormalise_value(self, input_value, mean, std):
        return input_value * std + mean


    def rolling_origin_forecast(self):
        
        if self.model == None:
            print('Fitting the model on validation data...')
            self.tune_model()
        print('starting rolling-origin forecast...')
        num_features = self.train_array.shape[1]
        #combining the train and validation arrays
        temp_train_arr = self.train_array.reshape(self.num_countries, -1, num_features)
        temp_val_arr = self.val_array.reshape(self.num_countries, -1, num_features)
        curr_train_arr = np.concat((temp_train_arr, temp_val_arr), axis = 1).reshape(-1, num_features)
        curr_test_arr = self.test_array
        # creating a dictionary for results
        forecasts_dict = {country: [] for country in self.list_of_countries}
        for i in range(self.test_array.shape[0] // self.num_countries):
            # normalising the train data 
            normalised_train_arr, mean_train, std_train = scaler(curr_train_arr)
            curr_train_inputs, curr_train_targets = generate_data_ffnn(normalised_train_arr, num_lags = self.num_lags, 
                                                                       num_countries = self.num_countries)
            
            # training the network on the normalised data
            self.model.fit(curr_train_inputs, curr_train_targets, batch_size = self.batch_size, epochs = self.epochs,
                           verbose = 0)

                    
            for j, country_name in enumerate(self.list_of_countries):
                # getting the last timestep data for current country
                temp_train_arr =  normalised_train_arr.reshape(self.num_countries, -1, num_features)
                forecast_input = temp_train_arr[j, -self.num_lags:, :].reshape(1, num_features * self.num_lags)
                # running inference
                model_forecast_value = self.model.predict(forecast_input, verbose = 0)[0,0]
                # unnormalising the value obtained from the model
                unnormalised_forecast_value = self.unnormalise_value(model_forecast_value, mean_train[-1], std_train[-1])
                # getting the original, undifferenced value of the forecast variable at the previous time step 
                # relative to the one being forecast
                if i == 0:
                    temp_original_train_arr = self.original_ro_train_array.reshape(self.num_countries, -1, num_features)
                    previous_value = temp_original_train_arr[j, -1, -1]
                else:
                    temp_original_test_arr = self.original_test_array.reshape(self.num_countries, -1, num_features)
                    previous_value = temp_original_test_arr[j, i - 1, -1]
                # undifferencing 
                reconstructed_forecast_value = unnormalised_forecast_value + previous_value
                # saving the forecast in its country's list in forecasts_dict
                forecasts_dict[country_name].append(reconstructed_forecast_value)
            # obtaining new training and test arrays by transferring the values already used for obtaining
            # forecast errors from test array to train array
            curr_test_arr = curr_test_arr.reshape(self.num_countries, -1, num_features)
            used_values = curr_test_arr[:, 0, :].reshape(self.num_countries, 1, num_features)
            curr_train_arr = curr_train_arr.reshape(self.num_countries, -1, num_features)
            curr_train_arr = np.concat((curr_train_arr, used_values), axis = 1).reshape(-1, num_features)
            curr_test_arr = curr_test_arr[:, 1:, :].reshape(-1, num_features)
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

