# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:02:56 2025
In this file, we compute the ANOVA statistic in Equation 4 from Worms, J., & Worms, R. (2015). A test for comparing tail indices for heavy-tailed distributions via empirical likelihood. Communications in Statistics-Theory and Methods, 44(15), 3289-3302.
@author: Anas Mourahib
"""


import numpy as np
import sys


import hill_estimator

from hill_estimator import Hill_estimator_fun
### data and extr_tail_fraction are lists of num_samples elements. the i element in data is the i-th sample and the i-th element of extr_tail_fraction is the proportion of extreme
### observations on the i-th sample
def statistic_fun (data : np.ndarray , extr_tail_fraction : np.ndarray   ) :
    num_samples = len(data)
    list = []
    for i in np.arange(num_samples):
        estim = Hill_estimator_fun(data[i] , extr_tail_fraction[i] , return_S = True)
        list.append(estim)
    ###caculus of gamma_Tilde
    vec_gamma = np.array([e[0] for e in list])
    vec_S_square = np.array( [e[1] for e in list] )
    vec_eta_square = np.array( [e[2] for e in list ] )
    gamma_tilde = sum(vec_eta_square * vec_gamma) / sum(vec_eta_square)
    
    ##Calculus of k
    k = np.array([e[3] for e in list])
    stat = sum(  k * pow(vec_gamma - gamma_tilde , 2 ) / pow(vec_S_square  , 0.5)  ) 
    return stat
###Example 1 : Two Frechet distributions 

N = 10000
X = -1 / np.log( np.random.uniform(0 , 1 , N) ) 
### Y  is a sample of Frechet(alpha) distribution
alpha = 2
sigma = 1
Y = sigma *  pow( -np.log( np.random.uniform(0 , 1 , N) ) , -1/alpha  )  

data = [X , Y]
extr_tail_fraction = [0.95 , 0.95]

#value_stat = statistic_fun(data , extr_tail_fraction)

#print("The value of the statistic for two Frechet distributions with same shape is ", value_stat)


