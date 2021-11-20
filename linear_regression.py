#!usr/bin/python3
# -*- coding : utf8 -*-

import numpy as np;


def cost_function(X, coef, y, scoring):
    """
        Compute the error committed by the linear
        model on the new coefficients.

        Parameters :
        -----------
        X : ndarray of shape (n_samples, n_features+1)
            Train data
        
        coef : ndarray of shape (n_features+1,)
            New coefficients

        y : ndarray of shape (n_samples,)
            True value
            
        scoring : str
            Name of the cost function

        Return :
        -------
        error : float
    """

    residuals = np.dot(X, coef) - y;

    if scoring == 'mse': # Mean Squared Error
        return np.sum(np.square(residuals)) / X.shape[0];
    
    elif scoring == 'rmse': # Root Mean Squared Error
        return np.sqrt(np.sum(np.square(residuals)) / X.shape[0]);
    
    elif scoring == 'mae': # Mean Absolute Error
        return np.sum(np.absolute(residuals)) / X.shape[0];
    
    else:
        raise ValueError("Something wrong with the cost function name.");


def gradient_vector(X, coef, y):
    """ 
        Calculate the gradient vector of the function to be minimized:
        (1/n) * sum_{i=1}^n (<X_i, coef> - y_i)^2  full batch
        (1/m) * sum_{i=1}^m (<X_i, coef> - y_i)^2  mini batch (m < n)
        (<X_i, coef> - y_i)^2  for stochastic
        
        Parameters :
        -----------
        X : ndarray of shape (n_rows, n_features+1)
            Train data on which we'll compute gradient
        
        coef : ndarray of shape (n_features+1,)
            Actual coefficients

        y : ndarray of shape (n_rows,)
            True value

        Return :
        -------
        gradient : ndarray of shape (n_features+1,)
    """

    y_pred = np.dot(X, coef);
    residuals = y_pred - y;
    
    return (2/X.shape[0]) * np.dot(X.T, residuals);


def check_batch(batch, n_samples):
    """ 
        Assure that batch isn't be larger than size of
        train data

        Parameters :
        -----------
        batch : int
            Size of mini train data to considered
        
        n_samples : int
            Size of train data
    """

    if batch > n_samples:
        raise ValueError(("'batch' is larger than the size"
                          " of the dataset (n_samples)."
                          " It must be less to n_samples."));


def idx_batch(batch, n_samples):
    """ 
        Get some indexes from train data
        
        Parameters :
        -----------
        batch : int
            Size of mini train data to considered
        
        n_samples : int
            Size of train data

        Return :
        -------
        array : ndarray of shape (batch, )
            Contains all of index of mini train
    """

    idx = [];
    repeat = 0;

    while repeat != batch:
        tmp = np.random.RandomState().randint(n_samples);

        if tmp not in idx:
            idx.append(tmp);
            repeat = repeat + 1;

    return np.array(idx);



class LinearRegression():
    """ 
        LinearRegression fits a linear model with coefficients to minimize
        the residual sum of squares between the observed targets in the
        dataset, and the targets predicted by the linear approximation.


        Parameters
        ----------
        n_iter : int, default=100
            Number of epochs to fit the model

        learning_rate : float, default=0.01
            Step size to control the rate of convergence
            
        batch : int, default=None
            If 'None' Vanilla method is used
            Else, either sgd, momentum or nesterov is used 
        
        momentum : float, default=None
            Accelarate the rate of convergence

        opt : {'gd', 'sgd', 'momentum', 'nesterov'}, default=None
            'gd' = Gradient Descent,
            'sgd' = Stochastic Gradient Descent,
            'momentum' = Momentum,
            'nesterov' = Nesterov Acceleration

        scoring : {'mse', 'rmse', 'mae'}, default='rmse'
            'mse' = Mean Squared Error,
            'rmse' = Root Mean Squared Error,
            'mae' = Mean Absolute Error


        Attributes
        ----------
        coef_ : ndarray of shape (n_features+1, )
            Estimated coefficients for the linear regression problem
            
        costs_ : ndarray of shape (n_iter, )
            Error committed during model fitting

        coef_history : ndarray of shape (n_iter, features+1)
            Coefficient of all epochs
    """
    
    def __init__(self, n_iter=100, learning_rate=0.01, batch=None, momentum=None, opt=None, scoring='rmse'):
        
        # Check 'n_iter'
        if isinstance(n_iter, int) is not True or n_iter <= 0:
            raise ValueError("'n_iter' should be an integer greater than 0."
                             " You  gave %s." % str(n_iter));

        # Check 'learning_rate'
        if learning_rate is not None and (isinstance(learning_rate, float) is not True\
                                     or learning_rate <= 0):
            raise ValueError("'learning_rate' should be an float greater than 0."
                             " You  gave %s." % str(learning_rate));
        if learning_rate is None:
            raise ValueError("'learning_rate' should be an float greater than 0."
                             " You  gave %s." % str(learning_rate));

        # Check 'batch'
        if batch is not None and (isinstance(batch, int) is not True or batch <= 0):
            raise ValueError("'batch' should be an integer greater than 0."
                             " You  gave %s." % str(batch));
        elif batch is not None and opt == 'gd':
            raise ValueError("'gd' don't need batch. Set 'batch' to None.");
        elif batch is None and opt in ['sgd', 'momentum', 'nesterov']:
            raise ValueError("If 'sgd', 'momentum' or 'nesterov' is"
                             " used, 'batch' don't must being to None.");

        # Check 'momentum'
        if momentum is not None and (isinstance(momentum, float) is not True or momentum <= 0):
            raise ValueError("'momentum' should be an float greater than 0."
                             " You  gave %s." % str(momentum));

        # Check 'opt'
        names_opt = ['gd', 'sgd', 'momentum', 'nesterov'];
        if opt is not None and opt not in names_opt:
            raise ValueError("'opt' should be an string."
                             " Either 'gd', 'sgd', 'momentum' or 'nesterov'."
                             " You  gave %s." % str(opt));

        # Check 'learning_rate' and 'opt'
        if learning_rate is not None and opt is None:
            raise ValueError("No optimization method was chosen."
                             " Choose one among ", names_opt);

        # Check 'opt' and 'momentum'
        if opt in ['momentum', 'nesterov'] and momentum is None:
            raise ValueError("'momentum' is None. Define it.");
        elif momentum is not None and opt in ['gd', 'sgd']:
            raise ValueError("'momentum' can only be defined for"
                             " 'momentum' and 'nesterov'. Set it to None");

        # Check 'scoring'
        names_scoring = ['mse', 'rmse', 'mae'];
        if scoring not in names_scoring:
            raise ValueError("'scoring' should be an string."
                             " Either 'mse', 'rmse' or 'mae'."
                             " You  gave %s." % str(opt));

        # Initialization
        self.n_iter = n_iter;
        self.learning_rate = learning_rate;
        self.batch = batch;
        self.momentum = momentum;
        self.opt = opt;
        self.scoring = scoring;


    def fit(self, X, y):
        """ 
            Fit the linear model

            Parameters
            ----------
                X : ndarray of shape (n_samples, n_features)
                    Training samples
                
                y : ndarray of shape (n_samples)
                    Target values
                
            Returns
            -------
                self
        """

        n_samples, n_features = X.shape;
        self.costs_ = np.empty((self.n_iter, ));
        self.coef_ = np.zeros((n_features+1, )); # +1 because we add intercept
        
        X = np.hstack((np.ones((n_samples, 1)), X)); # because we add intercept
        self.coef_history = np.empty((self.n_iter, n_features+1));

        if self.opt in ['momentum', 'nesterov']:
            vector = np.zeros((n_features+1,));

        for epoch in range(self.n_iter):
            
            if self.opt == 'gd':
                
                # GD Method
                gradient = gradient_vector(X, self.coef_, y);
                self.coef_ = self.coef_ - (self.learning_rate * gradient);

                # Cost
                self.costs_[epoch] = cost_function(X, self.coef_, y, self.scoring);
                self.coef_history[epoch] = self.coef_.T;

            if self.opt == 'sgd':

                check_batch(self.batch, n_samples);
                idx_samples = idx_batch(self.batch, n_samples);

                # SGD Method
                gradient = gradient_vector(X[idx_samples], self.coef_, y[idx_samples]);
                self.coef_ = self.coef_ - (self.learning_rate * gradient);
                
                # Cost
                self.costs_[epoch] = cost_function(X, self.coef_, y, self.scoring);
                self.coef_history[epoch] = self.coef_.T;

            if self.opt == 'momentum':
                
                check_batch(self.batch, n_samples);
                idx_samples = idx_batch(self.batch, n_samples);
                
                # Momentum Method 
                gradient = gradient_vector(X[idx_samples], self.coef_, y[idx_samples]);
                vector = (self.momentum * vector) + (self.learning_rate * gradient);
                self.coef_ = self.coef_ - vector;
                
                # Cost
                self.costs_[epoch] = cost_function(X, self.coef_, y, self.scoring);
                self.coef_history[epoch] = self.coef_.T;
                
            if self.opt == 'nesterov':
                
                check_batch(self.batch, n_samples);
                idx_samples = idx_batch(self.batch, n_samples);

                # Nesterov Method
                nestrov = self.coef_ - (self.momentum * vector);
                gradient = gradient_vector(X[idx_samples], nestrov, y[idx_samples]);
                vector = (self.momentum * vector) + (self.learning_rate * gradient);
                self.coef_ = self.coef_ - vector;
                   
                # Cost
                self.costs_[epoch] = cost_function(X, self.coef_, y, self.scoring);
                self.coef_history[epoch] = self.coef_.T;       


        return self;


    def predict(self, X):
        """ 
            Predicts the values with the trained model.
        
            Parameters
            ----------
                X : ndarray of shape (n_samples, n_features)
                    Test samples
            
            Returns
            -------
                y : ndarray of shape (n_samples, )
                    Predicted values
        """
        
        X = np.hstack( (np.ones((X.shape[0], 1)), X) );
        
        return np.dot(X, self.coef_);



        