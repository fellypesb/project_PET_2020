#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:29:59 2020

@author: fellypesb
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def evaluate_model(y_real, y_pred):
    mse = mean_squared_error(y_real, y_pred)
    rmse = mean_squared_error(y_real, y_pred, squared=False)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    return pd.DataFrame({'Metricas': ['mse','rmse','mae','r2'], 'Value': [mse, rmse, mae, r2]})
