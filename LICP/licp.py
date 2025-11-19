# Import necessary libraries
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import itertools
import copy
import matplotlib.pyplot as plt
import networkx as nx
import sys
from LICP.utils.helper import *
from LICP.mlmodel.predictor import TargetPrediction
import pandas as pd
import warnings




# Define the main class for Fault Detection



class pirca:
    def __init__(self, 
                model,
                lags=None,
                test_lags=None,
                progressbar=True):
        """
        Initializes the Fault Detection system.

        Usage of this can be found under "https://github.com/AlexanderMey/LICP_development/tree/main/examples"

        model: sklearn (default: LinearRegression)
            A machine learning model from sklearn that the user specifies.

        lags: dictionary
            A dictionary containing the lags we use to predict the future from the past. lags[cov][target] specifies the lag we use
            to predict target from cov.

        test_lags: list
            A list of potential lags we optimize over. Only used when lags==None.

        progressbar: bool
            Indicating if we want to plot the progressbar while computing.
                        
        """
        if test_lags and not lags:
            lags='Par_Corr'
        self.Predictor = TargetPrediction(model,lags,test_lags=test_lags)
        self.lags=lags
        self.test_lags=test_lags
        self.progress=0
        self.progressbar=progressbar
        self.graph={}
        self.model=model
        # Initilializing all the information we store to retreive later.
        self.overfit_warning={} #Saving warnings we output in case we overfitted. Also useful to not keep repeating for the same target
        self.plausible={}
        self.p_vals={}
        self.importance={}
        self.minimal={}
        self.Residuals_save={}
        self.parents={}
        self.fitness={}





    def compute_parents(self,
                        data, 
                        targets=[],
                        data_time=None,
                        causal_search='subsets',
                        subset_size=3,
                        fit_autoregression=False,
                        test='bootstrap',
                        ci='pivotal',
                        bootstrap=1000,
                        alpha=0.1,
                        frequency_match='target',
                        metric=np.var,
                       slack=0):
        """
        The main routine of our module. Computes parents of the input variables based on the specified criteria.

        data: list, entries of the list correspond to different environments in which the data was observed.

        targets: list, targets we want to find causal parents for. default: []
            If target==[] we loop over all variables defined as keys in data[x].
        
        data_time: list, timestamps of the observed data, with the same structure as data.
            If data_time=None it is assumed that the start and endpoints of the observed data is the same for all variables.

        causal_search: str, specifies how to identify causal parents of a variable. {'greedy','subsets'}), default: 'greedy'
            'greedy' adds and then removes greedily variables until no
            variable significantly increases the plausibility of the subset. 'subsets' loops over all subsets of size at most subset_size and
            computes afterwards minimal plausible sets based on the calculated scores.

        subset_size: int, specifying the maximal subset size of possible causal parents. default: 3
            Specifies the maximum size of the subset when looping over subsets of variables.

        fit_autoregression: bool, if True we also use the past of target to predict its future. default: False

        test: str, indicating which test to use to determine significance of plausible sets. {'bootstrap','permutation'}, default: 'bootstrap'

        ci: str, how to compute the confidence intervals. {'pivotal','quantile'}, default: 'pivotal'.

        bootstrap: int, amount of bootstrap repititions. default: 500

        alpha: float, the significance level we run the test with. default: 0.05

        frequency_match: str, indicates how to match time series of different frequencies. default: 'target'
            If the time series data has different length due to sampling at different frequencies, this specifies how to match
            the data for further prediction. 

        metric: mapping numpy array -> float, choice of the test metric. default: np.var
        """
        
        # Saving the arguments
        self.data=copy.deepcopy(data)
        self.targets=targets 
        self.causal_search=causal_search 
        self.bootstrap=bootstrap
        self.test=test
        self.ci=ci
        self.alpha=alpha
        self.metric=metric
        self.UFI_only=False
        # If the data is stored as numpy arrays we first transform it
        if not isinstance(self.data[0],dict):
            self.data=_data_transform(self.data)


        if data_time==None:
            data_time=[]
            for e in range(len(self.data)):
                data_time.append({})
                for cov in self.data[e]:
                    data_time[e][cov]=np.arange(0,1,1/len(self.data[e][cov]))


        if targets==[]:
            targets=[target for target in self.data[0]]
            self.targets=targets


        # The subset causal search for plausible subsets works in several steps.
        if self.causal_search=='subsets':
            print('Model Training Phase')
            for target in targets:
                d=len(targets)
                all_ind=[ind for ind in self.data[0]]

                if not fit_autoregression:
                    all_ind.remove(target)
                    
                # In case of time-series data with different frequencies we check if the 
                # samples match in length, and if not
                # up/downsample the data to match the target.
                self.data_matched=copy.deepcopy(self.data) 

                for e in range(len(self.data)):
                    for cov in self.data[e]:
                        if len(self.data[e][cov])!=len(self.data[e][target]):

                            self.data_matched[e][cov]=match_time_series(self.data[e][cov],data_time[e][cov],data_time[e][target])

 
                            
                if subset_size==None:
                    subset_size=d
                    warnings.warn('No maximal subset size provided, search is over all', 2**d  ,'subsets.')
                temp=[list(itertools.combinations(all_ind, k)) for k in range(0,subset_size+1)]
                subsets = [item for sublist in temp for item in sublist]
                subsets.append([])
                
                # 1. We loop over all subsets (ind) and compute the fitness of this subset to explain target
                for k,ind in enumerate(subsets):
                    self.compute_fitness(ind,target=target)
                    if self.progressbar:
                        self.update_progress(d*len(subsets))

    

                    
            
            # 2. We loop again over all targets to first find all plausible sets (based on the fitness) and then
            #    find which of those plausible sets are minimal. The fitness of the minimal sets is stored in self.parents
            self.progress=0
            print('\n Computing Plausible Sets')
            for key in self.fitness:      
                self._compute_plausible_sets_subsets(target=key,slack=slack)
                self.minimal[key]=[]
                self.parents[key]={}
                for ind in self.plausible[key]:
                    if is_minimal(ind,self.plausible[key]):
                        self.minimal[key].append(ind)
                        self._compute_importance(ind,key)
                        self.parents[key][ind]=self.fitness[key][ind]
                        
            self._compute_UFI()

        # If we use a greedy criterion to find plausible sets, we only compute the fitness of the subsets until we have
        # found a suitable set. For that reason there are no precomputations of the fitness.
        if self.causal_search=='greedy':
            print('Greedily finding parent sets of size maximal', subset_size)
            for target in targets:
                self.data_matched=copy.deepcopy(self.data) 

                for e in range(len(self.data)):
                    for cov in self.data[e]:
                        if len(self.data[e][cov])!=len(self.data[e][target]):
                            self.data_matched[e][cov]=match_time_series(self.data[e][cov],data_time[e][cov],data_time[e][target])
                self._compute_plausible_sets_greedy(self.data,target=target,subset_size=subset_size)
            if self.progress<len(targets)**2*subset_size:
                self.progress=len(targets)**2*subset_size-1
                self.update_progress(len(targets)**2*subset_size)
        
            self._compute_UFI()

    def compute_UFI(self,
                        data, 
                        targets=[],
                        data_time=None,
                        causal_search='subsets',
                        subset_size=3,
                        fit_autoregression=False,
                        metric=np.var):
        """
        Simplification of our main routine where we just compute the UFI to rank potential root causes.

        data: list, entries of the list correspond to different environments in which the data was observed.

        targets: list, targets we want to find causal parents for. default: []
            If target==[] we loop over all variables defined as keys in data[x].
        
        data_time: list, timestamps of the observed data, with the same structure as data.
            If data_time=None it is assumed that the start and endpoints of the observed data is the same for all variables.

        causal_search: str, specifies how to identify causal parents of a variable. {'greedy','subsets'}), default: 'greedy'
            'greedy' adds and then removes greedily variables until no
            variable significantly increases the plausibility of the subset. 'subsets' loops over all subsets of size at most subset_size and
            computes afterwards minimal plausible sets based on the calculated scores.

        subset_size: int, specifying the maximal subset size of possible causal parents. default: 3
            Specifies the maximum size of the subset when looping over subsets of variables.

        fit_autoregression: bool, if True we also use the past of target to predict its future. default: False

        metric: mapping numpy array -> float, choice of the test metric. default: np.var
        """
        
        self.UFI_only=True
        # Saving the arguments
        self.data=copy.deepcopy(data)
        self.targets=targets 
        self.causal_search=causal_search 
        self.metric=metric
        self.Predictor = TargetPrediction(self.model,self.lags,test_lags=self.test_lags,metric=self.metric)
        # If the data is stored as numpy arrays we first transform it
        if not isinstance(self.data[0],dict):
            self.data=_data_transform(self.data)


        if data_time==None and self.test_lags:
            warnings.warn('No timestamps provided. Calculating timestamps based on the assumptions that start and end time of all measurements are the same.')
            data_time=[]
            for e in range(len(self.data)):
                data_time.append({})
                for cov in self.data[e]:
                    data_time[e][cov]=np.arange(0,1,1/len(self.data[e][cov]))


        if targets==[]:
            targets=[target for target in self.data[0]]
            self.targets=targets


        # The subset causal search for plausible subsets works in several steps.
        if self.causal_search=='subsets':
            print('Model Training Phase')
            for target in targets:
                d=len(targets)
                all_ind=[ind for ind in self.data[0]]

                if not fit_autoregression:
                    all_ind.remove(target)
                    
                # In case of time-series data with different frequencies we check if the 
                # samples match in length, and if not
                # up/downsample the data to match the target.
                self.data_matched=copy.deepcopy(self.data) 

                for e in range(len(self.data)):
                    for cov in self.data[e]:
                        if len(self.data[e][cov])!=len(self.data[e][target]):

                            self.data_matched[e][cov]=match_time_series(self.data[e][cov],data_time[e][cov],data_time[e][target])

 
                            
                if subset_size==None:
                    subset_size=d
                    warnings.warn('No maximal subset size provided, search is over all', 2**d  ,'subsets.')
                temp=[list(itertools.combinations(all_ind, k)) for k in range(0,subset_size+1)]
                subsets = [item for sublist in temp for item in sublist]
                subsets.append([])
                
                # 1. We loop over all subsets (ind) and compute the fitness of this subset to explain target
                for k,ind in enumerate(subsets):
                    self.compute_fitness(ind,target=target)
                    if self.progressbar:
                        self.update_progress(d*len(subsets))
                        
            self._compute_UFI()

        # If we use a greedy criterion to find plausible sets, we only compute the fitness of the subsets until we have
        # found a suitable set. For that reason there are no precomputations of the fitness.
        if self.causal_search=='greedy':
            print('Greedily finding parent sets of size maximal', subset_size)
            for target in targets:
                self.data_matched=copy.deepcopy(self.data) 

                for e in range(len(self.data)):
                    for cov in self.data[e]:
                        if len(self.data[e][cov])!=len(self.data[e][target]):
                            self.data_matched[e][cov]=match_time_series(self.data[e][cov],data_time[e][cov],data_time[e][target])
                self._compute_plausible_sets_greedy(self.data,target=target,subset_size=subset_size,UFI_only=True)
            if self.progress<len(targets)**2*subset_size:
                self.progress=len(targets)**2*subset_size-1
                self.update_progress(len(targets)**2*subset_size)
        
            self._compute_UFI()
    
    def _compute_plausible_sets_subsets(self,target=None,slack=0):
        """
        Computes and stores all plausible sets for a chosen target variable based on precomputed scores.

        target: dictionary key, the variable we want to find causal parents for.
        """

        if target=='None':
            target=self.target

        self.plausible[target]=[]
        self.p_vals[target]={}
        for ind in self.fitness[target]:
            if not ind=='Max plausibility':
                if self.test=='bootstrap':
                        if self.progressbar:
                            self.update_progress(len(self.fitness)*(len(self.fitness[target])-1))
                        q_lower,_=self._compute_bootstrap_CI(target,tuple(sorted(ind)))
                        if q_lower<0:
                            self.plausible[target].append(ind)
                if self.test=='permutation':
                    p_val=self._compute_permutation_p_val(target,ind,slack=slack)
                    self.p_vals[target][ind]=p_val
                    if p_val>self.alpha:
                        self.plausible[target].append(ind)


    def _compute_plausible_sets_greedy(self,data,target,subset_size=3,UFI_only=False):
        """
        Greedily finds a plausible set of causal parents for target, with maximal set size subset_size.

        data: list, items correspond to dictionaries of observations in the environments.

        target: dictionary key, the target variable we want to find causal parents for.

        UFI_only: boolean, we only compute the UFI score to rank root causes, much faster.
        """
        if len(data)==0:
            data=copy.copy(self.data_matched)
        if target=='None':
            target=self.target
            


        n,d=data[0][target].shape
        all_targets_length=np.sum([data[0][target].shape[1] for target in self.targets])
        all_ind=[ind for ind in data[0]]
        all_ind.remove(target)
        
        for o in range(d):
            if d==1:
                key=target
            else:
                key=(target,o)

            left_indices=copy.copy(all_ind) # Indices we did not add yet in a greedy step

            self.compute_fitness([],target=target) # Defines our baseline score

            greedy_indices=[]

            # We first perform a greedy forward pass, adding indices until we cannot significantly 
            # increase the score or the maximal subset_size is reached.
            while len(left_indices)>0 and len(greedy_indices)<subset_size:
                greedy_update={} 
                for i in left_indices:
                    ind=copy.copy(greedy_indices)
                    ind.append(i)
                    if self.progressbar:
                        self.update_progress(2*all_targets_length*subset_size*len(data[0]))
           
                    if tuple(sorted(ind)) not in self.fitness[key]:
                        self.compute_fitness(tuple(sorted(ind)),target=target)
                    # print('forward',ind,target) 
                    if key not in greedy_update:
                        
                        greedy_update[key]={}
                        greedy_update[key]['value']=self.fitness[key][tuple(sorted(ind))]
                        greedy_update[key]['index']=i
                            
                    else:
                        if self.fitness[key][tuple(sorted(ind))]>greedy_update[key]['value']:
                            greedy_update[key]['value']=self.fitness[key][tuple(sorted(ind))]
                            greedy_update[key]['index']=i
                greedy_indices_comp=copy.copy(greedy_indices)
                greedy_indices.append(greedy_update[key]['index'])
                
                if not UFI_only:
                    q_lower,q_upper=self._compute_bootstrap_CI(key,tuple(sorted(greedy_indices)),ind_comp=tuple(sorted(greedy_indices_comp)))
                    if 0<q_lower:
                        greedy_indices.remove(greedy_update[key]['index'])
                        break
                        
                    else:
                        left_indices.remove(greedy_update[key]['index'])
                
                if UFI_only:
                    if self.fitness[key][tuple(sorted(greedy_indices))]>=self.fitness[key][tuple(sorted(greedy_indices_comp))]:
                        greedy_indices.remove(greedy_update[key]['index'])
                        break
                    else:
                        left_indices.remove(greedy_update[key]['index'])

            # We then perform a greedy backward pass to see if we can eliminate any variables from the subset found in the
            # greedy forward pass
            if not UFI_only:
                while len(greedy_indices)>0:
                    greedy_update={} 
                    for i in greedy_indices:
                        ind=copy.copy(greedy_indices)
                        ind.remove(i)
                        self.update_progress(2*all_targets_length*subset_size*len(data[0]))

                        if set(ind) not in [set(key1) for key1 in self.fitness[key]]:
                            self.compute_fitness(tuple(sorted(ind)),target=target)
                    
                        if key not in greedy_update:
                            
                            greedy_update[key]={}
                            greedy_update[key]['value']=self.fitness[key][tuple(sorted(ind))]
                            greedy_update[key]['index']=i
                                
                        else:
                            if self.fitness[key][tuple(sorted(ind))]>greedy_update[key]['value']:
                                greedy_update[key]['value']=self.fitness[key][tuple(sorted(ind))]
                                greedy_update[key]['index']=i
                    greedy_indices_comp=copy.copy(greedy_indices)
                    greedy_indices.remove(greedy_update[key]['index']) 
                    q_lower,q_upper=self._compute_bootstrap_CI(key,tuple(sorted(greedy_indices)),ind_comp=tuple(sorted(greedy_indices_comp)))
                    if 0<q_lower:
                        self.parents[key]={}
                        self.parents[key][tuple(sorted(greedy_indices_comp))]=self.fitness[key][tuple(sorted(greedy_indices_comp))]
                        self._compute_importance(tuple(sorted(greedy_indices_comp)),key)
                        self.minimal[key]={}
                        self.minimal[key]=[tuple(sorted(greedy_indices_comp))]
                        break
    
    def compute_fitness(self,ind,target=None):
        """
        Computes and stores plausibility (fitness) score for a given target and a set of indices.

        ind: tuple, subset of indices.
        data: list, the data we base our decision on, usually inherited from the main class.
        target: dictionary key, the target we want to find causal parents for
        """

        if target=='None':
            target=self.target
   
        y=[] # Stores the data of the target variable
        E=len(self.data_matched)
        Res=[] # Stores residuals which we use for final computations

        if self.metric==np.mean:
            env_sample_length=[]
            env_sample_length.append(0)
        for e in range(E):
            # Making sure y has the right numpy shape, note it can be multivariate.
            if np.prod((self.data_matched[e][target]).shape) == (self.data_matched[e][target]).shape[0] and not isinstance(self.Predictor.model, MultiOutputRegressor):
                y.append(np.ravel(self.data_matched[e][target]))


            else:
                y.append((self.data_matched[e][target]))
            
            if self.metric!=np.mean:
                if len(ind)==0:
                    Res.append(y[e]-np.mean(y[e],axis=0))


                else:
                    # Getting the residuals when fitting the data in environment e of indices to the
                    # data of target.
                    Residual=self.Predictor.get_residuals(self.data_matched[e],ind,target)
                    Res.append(Residual)

            else:
                env_sample_length.append((self.data_matched[e][target]).shape[0])
        
        if self.metric==np.mean:
            if len(ind)==0:
                    for e in range(E):
                        Res.append(y[e])
            else:
                data_global={}
                for cov in self.data_matched[0]:
                    data_global[cov]=np.concatenate([self.data_matched[e][cov] for e in range(E)])
                Residual=self.Predictor.get_residuals(data_global,ind,target)

                for e in range(E):
                    Res.append(Residual[env_sample_length[e]:env_sample_length[e]+env_sample_length[e+1]])
    
                    
        # After we computed the residuals, we now compute a fitness score based on those
        n_min=min([len(Res[e]) for e in range(E)])
        Res=[Res[e][:n_min] for e in range(E)]

        if self.metric!=np.mean:
            if np.round(np.max(self.metric(np.asarray(Res),axis=1),axis=0),10).all()==0:
                if target not in self.overfit_warning:
                    name=str(target)
                    warnings.warn('\n Vanishing residuals for variable ' +target+ ' as target. Parent sets are untrustworthy.')
                    self.overfit_warning[target]=True

        if self.metric!=np.mean:
            statavg=np.min(self.metric(np.asarray(Res),axis=1),axis=0)/np.max(self.metric(np.asarray(Res),axis=1),axis=0)
        else:
            statavg=1/np.exp(np.max(self.metric(np.asarray(Res),axis=1),axis=0)-np.min(self.metric(np.asarray(Res),axis=1),axis=0))


        if isinstance(statavg,float):
            length=1
            statavg=[statavg]
        else:
            length=len(statavg)

        for o in range(len(statavg)):
            if length==1:
                key=target
            else:
                key=(target,o)
            if key not in self.fitness:
                self.fitness[key]={}
                self.Residuals_save[key]={}
            if length==1:
                self.Residuals_save[key][tuple(sorted(ind))]=[Res[e][:n_min] for e in range(E)]
            else:
                self.Residuals_save[key][tuple(sorted(ind))]=[Res[e][:n_min,o] for e in range(E)]
            self.fitness[key][tuple(sorted(ind))]=statavg[o]
            

            if 'Max plausibility' not in self.fitness[key]:
                self.fitness[key]['Max plausibility']=(tuple(sorted(ind)),statavg[o])
            else:
                if self.fitness[key]['Max plausibility'][1]<statavg[o]:
                    self.fitness[key]['Max plausibility']=(tuple(sorted(ind)),statavg[o])



    def _compute_importance(self,ind,target):
        """
        Computes the importance of individual variables within plausible sets. Importance is measured as the ratio
        of the fit with and without the individual variable.

        ind: dictionary key, the variable we want to compute the importance for.
        target: dictionary key, the target that the variable should explain.
        """
        if target not in self.importance:
            self.importance[target]={}
        for var in ind:
            if var not in self.importance[target]:
                self.importance[target][var]=0
            ind_rem=copy.copy(list(ind))
            ind_rem.remove(var)
            ind_rem=tuple(ind_rem)
            if set(ind) not in [set(key) for key in self.fitness[target]]:
                self.compute_fitness(ind,target=target)
            if set(ind_rem) not in [set(key) for key in self.fitness[target]]:
                self.compute_fitness(ind_rem,target=target)
            # self.importance[target][var]=min([self.importance[target][var],
            #                                     1-(self.fitness[target][ind_rem])/(self.fitness[target][ind])])                                                                      
            
            self.importance[target][var]=max([self.importance[target][var],
                                                (self.fitness[target][ind])-(self.fitness[target][ind_rem])])                                                                      
    
    def _compute_UFI(self):
        """
        Computes the UFI score based on the computed fitness
        """
        self.UFI={}
        for key in self.fitness:
            if isinstance(key,tuple):
                target=key[0]
            else:
                target=key
            if target not in self.UFI:
                self.UFI[target]=1-self.fitness[key]['Max plausibility'][1]
            else:
                self.UFI[target]=max([self.UFI[target],1-self.fitness[key]['Max plausibility'][1]])

    def _compute_bootstrap_CI(self,target,ind,ind_comp=False):
        """
        Computes the lower confidence region of the plausibility of the indices 'ind' for the target. For that we need a comparison
        set of (currently) best indicices, which are stored in self.fitness[target]['Max plausibility']

        target: dictionary key, the target we want to find plausible sets for.
        ind: tuple, the indices we check for plausibility.
        ind_comp: tuple, indices we compare ind against to check for plausibility. default: False, in that case we take the best indices as comparison.

        returns: q_lower,q_upper; float,float: the lower and upper confidence level for the computed statistic. 
        """


        B=self.bootstrap
        E=len(self.Residuals_save[target][ind])
        # If we selected different lags in the environments we also get different length of residuals. We then make sure the legth matches.
        n_min=min([len(self.Residuals_save[target][ind][e]) for e in range(E)])
        Res=[self.Residuals_save[target][ind][e][:n_min] for e in range(E)]
        indices=np.random.choice(np.arange(0,n_min,1), size=(n_min,B,1), replace=True, p=None)

        boots_residuals=[Res[e][indices] for e in range(E)]
        if ind_comp or ind_comp==():
            ind_best=ind_comp
        else:
            ind_best=self.fitness[target]['Max plausibility'][0]
        n_min=min([len(self.Residuals_save[target][ind_best][e]) for e in range(E)])
        Res_best=[self.Residuals_save[target][ind_best][e][:n_min] for e in range(E)]
        indices=np.random.choice(np.arange(0,n_min,1), size=(n_min,B,1), replace=True, p=None)

        boots_residuals_best=[Res_best[e][indices] for e in range(E)]
        if np.round(np.max(self.metric(np.asarray(Res),axis=1),axis=0),10).all()==0 and self.metric!=np.mean:
            bootsstats=np.ones((42,1))
            if target not in self.overfit_warning:
                name=str(target)
                warnings.warn('\n Vanishing residuals for variable ' +name+ ' as target. Parent sets are untrustworthy.')
                self.overfit_warning[target]=True
            q_lower=np.ones((1,1))
            q_upper=np.ones((1,1))
            
        else:
            # Stores all test statistics after bootstrapping
            e_min=np.argmin(self.metric(np.asarray(Res),axis=1),axis=0)
            e_max=np.argmax(self.metric(np.asarray(Res),axis=1),axis=0)
            e_min_best=np.argmin(self.metric(np.asarray(Res_best),axis=1),axis=0)
            e_max_best=np.argmax(self.metric(np.asarray(Res_best),axis=1),axis=0)

            if self.metric!=np.mean:
                bootsstats=self.metric(boots_residuals_best[e_min_best],axis=0)/self.metric(boots_residuals_best[e_max_best],axis=0) - \
                        self.metric(boots_residuals[e_min],axis=0)/self.metric(boots_residuals[e_max],axis=0) 
                
                statavg=self.metric(Res_best[e_min_best])/self.metric(Res_best[e_max_best])- \
                        self.metric(Res[e_min])/self.metric(Res[e_max])
            
            else:
                bootsstats=1/np.exp(self.metric(boots_residuals_best[e_max_best]-self.metric(boots_residuals_best[e_min_best],axis=0),axis=0)) - \
                        1/np.exp(self.metric(boots_residuals[e_max],axis=0)-self.metric(boots_residuals[e_min],axis=0)) 
                
                statavg=1/np.exp(self.metric(Res_best[e_max_best])-self.metric(Res_best[e_min_best]))- \
                        1/np.exp(self.metric(Res[e_max])-self.metric(Res[e_min]))
                

            if isinstance(statavg,float):
                length=1
            else:
                length=len(statavg)
            if self.ci=='pivotal':
                q_lower=np.reshape(2*statavg-np.quantile(bootsstats,1-self.alpha/2,axis=0),(length,1))
                q_upper=np.reshape(2*statavg-np.quantile(bootsstats,self.alpha/2,axis=0),(length,1))
    
            if self.ci=='quantile':
                q_lower=np.reshape(np.quantile(bootsstats,self.alpha/2,axis=0),(length,1))
                q_upper=np.reshape(np.quantile(bootsstats,1-self.alpha/2,axis=0),(length,1))
        return q_lower,q_upper

    def _compute_permutation_p_val(self,target,ind,slack=0):
        """
        Computes the lower confidence region of the plausibility of the indices 'ind' for the target. For that we need a comparison
        set of (currently) best indicices, which are stored in self.fitness[target]['Max plausibility']

        target: dictionary key, the target we want to find plausible sets for.
        ind: tuple, the indices we check for plausibility.
        ind_comp: tuple, indices we compare ind against to check for plausibility. default: False, in that case we take the best indices as comparison.

        returns: q_lower,q_upper; float,float: the lower and upper confidence level for the computed statistic. 
        """


        B=self.bootstrap
        E=len(self.Residuals_save[target][ind])
        # If we selected different lags in the environments we also get different length of residuals. We then make sure the legth matches.
        n_min=min([len(self.Residuals_save[target][ind][e]) for e in range(E)])
        Res=[self.Residuals_save[target][ind][e][:n_min] for e in range(E)]
        Res_stack=np.hstack([self.Residuals_save[target][ind][e][:n_min] for e in range(E)])
        indices=np.random.choice(np.arange(0,E*n_min,1), size=(E*n_min,B,1), replace=True, p=None)

        permutation_residuals=[Res_stack[indices[e*n_min:(e+1)*n_min,:]] for e in range(E)]




        if np.round(np.max(self.metric(np.asarray(Res),axis=1),axis=0),10).all()==0:
            bootsstats=np.ones((42,1))
            if target not in self.overfit_warning:
                name=str(target)
                warnings.warn('\n Vanishing residuals for variable ' +name+ ' as target. Parent sets are untrustworthy.')
                self.overfit_warning[target]=True
            p_val=1
            
        else:
            # Stores all test statistics after bootstrapping
            e_min=np.argmin(self.metric(np.asarray(Res),axis=1),axis=0)
            e_max=np.argmax(self.metric(np.asarray(Res),axis=1),axis=0)
            bootsstats=self.metric(permutation_residuals[e_min],axis=0)/self.metric(permutation_residuals[e_max],axis=0)
            
        
            statavg=self.metric(Res[e_min])/self.metric(Res[e_max])
            counts=[statavg>=bootsstats-slack]
            if isinstance(statavg,float):
                length=1
            else:
                length=len(statavg)
     
            p_val=np.reshape(np.sum(counts),(length,1))/B


        return p_val

    def visualize(self,fit_min=0,importance_min=0,edge_score='importance',save=False):
        """
        Visualizes the causal graph using matplotlib and networkx.

        fit_min: float, between 0 and 1. We visiualize only edges with a fitness of at least fit_min. default: 0
        importance_min: float, between 0 and 1. We visiualize only edges with an importance of at least importance_min. default: 0
        """
        self.vis_graph = nx.DiGraph()
        for target in self.minimal:
            for ind in self.minimal[target]:
                if not len(ind)==0:
                    if isinstance(target,tuple):
                        key=target[0]
                    else:
                        key=target
                    for var in ind:

                        if not self.vis_graph.has_edge(var,key):

    
                            fit=(self.parents[target][ind])
                            impact=self.importance[target][var]
                            if fit>fit_min and impact>importance_min:
                                fit=round(fit,2)
                                impact=round(impact,2)
                                if edge_score=='importance':
                                    self.vis_graph.add_edge(str(var)+'\n'+str(round(self.UFI[var],2)), str(key)+'\n'+str(round(self.UFI[key],2)), weight=impact,width=1, color='green')
                                if edge_score=='fitness':
                                    self.vis_graph.add_edge(str(var)+'\n'+str(round(self.UFI[var],2)), str(key)+'\n'+str(round(self.UFI[key],2)), weight=fit,width=1, color='green')
                        else:
        
                            fit=(self.parents[target][ind])
                            impact=self.importance[target][var]
                            prev_score=self.vis_graph[var][key]['weight']
                            fit=round(fit,2)
                            impact=round(impact,2)
                            if edge_score=='importance':
                                if impact>prev_score:
                                    self.vis_graph.remove_edge(str(var)+'\n'+str(round(self.UFI[var],2)), str(key)+'\n'+str(round(self.UFI[key],2)))
                                    self.vis_graph.add_edge(str(var)+'\n'+str(round(self.UFI[var],2)), str(key)+'\n'+str(round(self.UFI[key],2)), weight=impact,width=1, color='green')
                            if edge_score=='fitness':
                                if fit>prev_score:
                                    self.vis_graph.remove_edge(str(var)+'\n'+str(round(self.UFI[var],2)), str(key)+'\n'+str(round(self.UFI[key],2)))
                                    self.vis_graph.add_edge(str(var)+'\n'+str(round(self.UFI[var],2)), str(key)+'\n'+str(round(self.UFI[key],2)), weight=fit,width=1, color='green')

                                

        
        
        # Extract edges and their attributes for easy access
        edges = self.vis_graph.edges(data=True)

        # node_labels = {node: node for node in G.nodes()}
        node_labels = {node: node for node in self.vis_graph.nodes()}
        # node_sizes = [len(node) * 500 for node in self.vis_graph.nodes()]
        pos = nx.spring_layout(self.vis_graph)  # positions for all nodes
        # x_values, y_values = zip(*pos.values())
        # x_min, x_max = min(x_values), max(x_values)
        # y_min, y_max = min(y_values), max(y_values)

        # Calculate width and height of the plot
        # x_range = x_max - x_min
        # y_range = y_max - y_min

        # # Add some padding around the nodes
        # padding = 0.1  # 10% padding
        # x_padding = x_range * padding
        # y_padding = y_range * padding

        # # Set figure size based on the range and padding
        # fig_width = (x_range + 2 * x_padding) * 5
        # fig_height = (y_range + 2 * y_padding) * 5
        plt.figure()

        
        # Draw nodes
        nx.draw_networkx_nodes(self.vis_graph, pos, node_color='skyblue')
  
        # Draw edges with colors and widths
        nx.draw_networkx_edges(self.vis_graph, pos, edgelist=edges,width=[d['width'] for u, v, d in edges],edge_color=[d['color'] for u, v, d in edges],connectionstyle="arc3,rad=0.2",min_target_margin=20)
        

        nx.draw_networkx_labels(self.vis_graph, pos,labels=node_labels, font_size=14, font_family='sans-serif')
        edge_labels = nx.get_edge_attributes(self.vis_graph, 'weight')
        my_draw_networkx_edge_labels(self.vis_graph, pos,edge_labels=edge_labels,rotate=True,rad = 0.2)

        # plt.xlim(x_min - x_padding, x_max + x_padding)
        # plt.ylim(y_min - y_padding, y_max + y_padding)
        if save:
            plt.savefig("Graph.pdf",format="pdf",bbox_inches='tight')
        plt.axis('off')  
        plt.show()

    def update_progress(self,total):

        self.progress+=1
        percent = 100 * (self.progress / float(total))
        bar = '█' * int(percent) + '-' * (100 - int(percent))
        sys.stdout.write("\r|{bar}| {percent:.2f}%".format(bar=bar, percent=percent))
        sys.stdout.flush()



