# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:50:24 2025

Here, we simulate from a mixture of Gaussian and Pareto

@author: Anas Mourahib
"""

import numpy as np
from scipy.stats import t

def generation (sizes : list, coef : list, linear_coef : list): 
    ##In the following num_e denotes the number of envi 
    ##size is a list of num_e elements and the e element contains the sample size in environemnt e 
    ##coef is a list of num_e matrices. Each matrix has ( (S+1) * 2) elements. The first colmn is the scale and the second is the shape
    ##linear_coef is a list of num_e arrays. Each array is of length S containing $\beta^e_S$ 
    num_env = len(sizes)
    list = []
    
    for e in np.arange(num_env):
        n_e = sizes[e] 
        coef_scale_shape = coef[e]
        slopes_e = np.array(linear_coef[e])
        print("This is slopes" , slopes_e)
        num_covariates = len(slopes_e) 
        covariates_e = np.zeros((n_e , num_covariates))
        for j in np.arange( num_covariates ):
            scale_j = coef_scale_shape[j][0]
            shape_j = coef_scale_shape[j][1]
            X_j =  t.rvs(df = shape_j, loc = 0 , scale = scale_j , size = n_e) 
            covariates_e[ : , j ] = X_j
        noise_e = t.rvs(df = coef_scale_shape[num_covariates][1], loc = 0 , scale = coef_scale_shape[num_covariates][0], size = n_e)
        Y_e = covariates_e @ slopes_e + noise_e 
        list.append(np.column_stack((covariates_e ,  noise_e , Y_e)))
        
    return list


## Example with two envirnements, 
#sizes = [10 , 10]
#coef_r = [[2 , 5] , [3 , 6] ,[3 , 10]]  # coefficients for regular enviroenement
#coef_f = [[5 , 1] , [3 , 1.5] ,[3 , 10]] # coefficients for faulty  enviroenement 

#coef = [coef_r , coef_f] 

#linear_coef = [[1,2] , [2,1]]

#sample = generation(sizes, coef , linear_coef)
