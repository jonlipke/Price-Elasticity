Overview

This repository demonstrates two approaches for measuring price elasticity using Python: Traditional elasticity calculation and Regression-based elasticity modeling.

Both approaches use sample pricing and volume data to estimate how demand responds to price changes.

Method 1 — Traditional Elasticity Calculation
The traditional approach calculates elasticity using period-to-period percentage changes.

Steps
1. Sort observations by date
2. Identify previous price and volume
3. Calculate percentage change in price
4. Calculate percentage change in volume
5. Compute elasticity per observation
6. Calculates a volume-weighted average elasticity

Method 2 — Regression Elasticity Model
The regression model estimates elasticity using a log-log demand model.

The regression workflow includes:
1. Log transformation of price and demand
2. Seasonal month dummy variables
3. Demand forecasting
4. Model validation to detect overfitting

Outputs
1. Regression output includes historical observations, forecasted volumes, and calcualted revenue
2. Regression results includes model coefficients, p-values, R², adjusted R², and RMSE
3. Regression model validation includes training RMSE, test RMSE, RMSE ratio, training/test observation counts

Dependencies: pip install pandas numpy statsmodels
