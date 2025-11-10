# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:58:26 2025
Here, we encode the Hill-estimator (HILL, B. M. (1975). A simple general approach to inference about the tail of a distribuition. Ann.
Statist. 3 1163-1174.) for the shape parameter of an extreme value distribution
@author: Anas Mourahib
"""
import numpy as np
import statistics
def Hill_estimator_fun(data : np.ndarray , extr_tail_fraction : np.ndarray , return_S : bool = False  ) -> np.ndarray:
    threshold = np.quantile(data , q = extr_tail_fraction)
    exceedances = data[data > threshold]
    k = len(exceedances)
    sorted_exceedances = sorted(exceedances)
    Y = []
    for i in np.arange(k-1):
        log_spacing_i = (i+1) * np.log(sorted_exceedances[k-1-i] / sorted_exceedances[k-1-i-1])
        Y.append( log_spacing_i)
    gamma_hat = statistics.mean(Y)
    if not return_S :
        return gamma_hat
    Y = np.array(Y)
    S_square__gammahat = statistics.mean(pow(Y-gamma_hat  , 2))
    eta_square = k/S_square__gammahat
    return gamma_hat , S_square__gammahat, eta_square, k
    
##### Example with Frechet distribution
#N = 10000
#X = -1 / np.log( np.random.uniform(0 , 1 , N) ) 



#gamma_hat = Hill_estimator_fun(X, 0.95)

#print("The Hill estimator of the shape parameter of a unit Frechet distribution is ", gamma_hat)
          
    
    
