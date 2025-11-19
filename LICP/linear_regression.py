# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:25:20 2025

@author: 20254817
"""
import numpy as np 
from sklearn import linear_model

##Here sample is a numpy array of shape n^e * (p+2). Each row is an observation. The first p columns are covariates, the (p+1) column is the noise and the 
### (p+2) column is a response
### The function linear_regression returns the estimates and the residuals of the linear regression of Y on the covariates
def linear_regression_fun (sample : np.ndarray , S : np.ndarray) -> list : 
    num_covariates = sample.shape[1]
    indx_covariates = S
    covariates = sample [: , indx_covariates]
    Y = sample[: , num_covariates - 1]
    print(Y)
    regr = linear_model.LinearRegression()
    regr = regr.fit(covariates , Y)
    Y_hat =  (covariates * regr.coef_).sum(axis=1)
    resid = Y - Y_hat
    return   Y_hat , resid


