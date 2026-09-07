## Forecasting Unemployment Rate with Econometric Models and Neural Networks 
#### By Illia Zhupanov

📄 [Read the full paper](research_paper/your_filename.pdf)

This repository contains the code, data, and the resulting forecasts for my research paper, which compares the performance of ARIMAX against deep-learning neural networks (FFNN and RNN) in forecasting the unemployment rate in OECD economies, with a COVID-recession robustness test. The naive baseline is the no-change model.



### Result

RNN and ARIMAX beat the naive baseline significantly pre-COVID; during the recession, none of the models could.



### Why unemployment rate

Motivated by the Okun's Law and the Phillips curve; RGDP growth and CPI inflation were used as input features/exogenous regressors.



### Data

A 22-country quarterly panel of OECD economies, from 1994 to 2019 for the main test; until 2023 for the COVID test, with Japan excluded due to data discontinuity.

Features: quarterly unemployment rate; quarterly CPI year-on-year inflation; quarterly RGDP growth



### Models

* **Naive** - no-change baseline
* **FFNN and RNN** - trained across the full panel, hyperparameters selected via grid search.
* **ARIMAX** - fitted per country, orders selected via grid search per country around auto-ARIMA suggestions.



### Evaluation

All models were evaluated using the rolling-origin expanding-window forecasting, refitted at every origin. Compared on MAE/RMSE and the Diebold-Mariano test.



### Repo structure:

* research\_code\_and\_data - models code, forecasting notebooks, data, and raw results
* research\_paper/ — full PDF write-up 
* .gitignore

