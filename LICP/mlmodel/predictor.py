import numpy as np
from LICP.utils.helper import calculate_partial_correlation
from sklearn.multioutput import MultiOutputRegressor
import copy


class TargetPrediction:
    def __init__(self,model,lags=None,test_lags=None,metric=np.var):
        """
        Initializes the model that predicts the target variable based on observed covarites.
        :param model: A machine learning model to predict the target. Currently we can use models from sklearn or darts
        """
        self.model=model
        if getattr(model, '__module__', '').startswith('sklearn'):
            self.type='sklearn'
        elif getattr(model, '__module__', '').startswith('darts'):
            self.type='darts'
        if lags=='Par_Corr':
            self.lags='Par_Corr'
            self.lag_matrix={}
            self.lag_values={}
        else:
            self.lags=None
        self.test_lags=test_lags
        self.metric=metric

    def get_residuals(self,data,ind,target):
        """
        Predict the target based on X_train with a fitted model

        :param X_train: Training data features.
        """
    
        if np.prod((data[target]).shape) == (data[target]).shape[0] and not isinstance(self.model, MultiOutputRegressor):
            y_train=np.ravel(data[target])
     
        else:
            y_train=data[target]
        if self.type=='sklearn':
            if self.lags=='Par_Corr':
                lags_temp=[]
                for cov in ind:
                    ind_temp=copy.copy(list(ind))
                    ind_temp.remove(cov)
                    if len(ind_temp)>0:
                        cond=copy.copy(np.concatenate([data[mod] for mod in ind_temp],axis=1))
                    else:
                        cond=[]
                    par_cor=calculate_partial_correlation(data[cov],y_train,self.test_lags,subset=cond)
                    self.lag_matrix[(cov,target,ind)]=self.test_lags[np.unravel_index(np.argmax(np.abs(par_cor)), par_cor.shape)[0]]
                    self.lag_values[(cov,target,ind)]=par_cor
                    lags_temp.append(self.lag_matrix[(cov,target,ind)])
                self.start=max(lags_temp)
            elif isinstance(self.lags,dict):
                self.start=max(self.lags[target].values())
            self.fit(data,ind,y_train,target)
            predictions=self.predict(data,ind,target)
            if self.lags==None:
                Residuals=y_train-predictions
            else:

            
                Residuals=y_train[self.start:]-predictions
            if self.metric==np.mean:
                try:
                    Residuals=Residuals-self.model.intercept_
                except:
                    print('Your model does not have a specified intercept at model.intercept_. This is needed for the mean metric')
            return Residuals
            
        elif self.type=='darts':
            return self.predictions.values()
    
    def fit(self,data,ind,y_train,target):
        """
        Fit the specified machine learning model to the training data.
        
        :param X_train: Training data features.
        :param y_train: Training data labels.
        """
        if self.lags==None:
            x_train=np.concatenate([data[mod] for mod in ind],axis=1)
        elif isinstance(self.lags,str):
            x_train=np.concatenate([data[mod][self.start-self.lag_matrix[(mod,target,ind)]:-self.lag_matrix[(mod,target,ind)]] for mod in ind],axis=1)
            y_train=y_train[self.start:]
        elif isinstance(self.lags,dict):
            x_train=np.concatenate([data[mod][self.start-self.lags[target][mod]:-self.lags[target][mod]] for mod in ind],axis=1)
            y_train=y_train[self.start:]
            print(self.start)
        self.model.fit(x_train, y_train)

        # elif self.type=='darts':
        #     x_train={mod:data[mod] for mod in ind}
        #     y_train_darts=dict_to_timeseries(data[target])
        #     x_train_darts=dict_to_timeseries(x_train)
        #     start_pred=3
        #     self.predictions=self.model.historical_forecasts(y_train_darts, past_covariates=x_train_darts,start=start_pred)

    def predict(self,data,ind,target):
        """
        Predict the target based on X_train with a fitted model

        :param X_train: Training data features.
        """
        if self.type=='sklearn':
            if self.lags==None:
                x_train=np.concatenate([data[mod] for mod in ind],axis=1)
            else:
                x_train=np.concatenate([data[mod][self.start-self.lag_matrix[(mod,target,ind)]:-self.lag_matrix[(mod,target,ind)]] for mod in ind],axis=1)
            
            return self.model.predict(x_train)
            
        elif self.type=='darts':
            return self.predictions.values()
    