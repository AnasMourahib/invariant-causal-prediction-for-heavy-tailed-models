# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 18:09:00 2025

@author: Anas Mourahib
"""

import statistic

from statistic import statistic_fun

import numpy as np
from sklearn import linear_model


N = 10000
Beta1 = 1 
Beta2 = 2

###Environement r 
X1_r = 1 + np.random.pareto(1, N)  
X2_r = 1 + np.random.pareto(1, N) 
epsilon_r = 1 + np.random.pareto(2, N) 
Y_r = Beta1 * X1_r + Beta2 * X2_r + epsilon_r 
covariates_r  = np.column_stack((X1_r, X2_r))
###Environement f 
X1_f = 1 +  np.random.pareto(1, N)  
X2_f = 1 + np.random.pareto(1, N) 
epsilon_f = (np.random.pareto(2, N) + 1 ) * 2
Y_f = Beta1 * X1_f + Beta2 * X2_f + epsilon_f 
covariates_f  = np.column_stack((X1_f, X2_f))



####For environement r
regr = linear_model.LinearRegression()
regr.fit(covariates_r , Y_r)
Y_r_hat =  (covariates_r * regr.coef_).sum(axis=1)
resid_r = Y_r - Y_r_hat
###For environement f
regr = linear_model.LinearRegression()
regr.fit(covariates_f , Y_f)
Y_f_hat =  (covariates_f * regr.coef_).sum(axis=1)
resid_f = Y_f - Y_f_hat


data = [resid_r , resid_f]
extr_tail_fraction = [0.95 , 0.95]

value_stat = statistic_fun(data , extr_tail_fraction)

print("The value of the statistic for the residuals of the twon environements is given by ", value_stat)