# This file allows users to run an experimental comparison between my own fast gradient 
# algorithm and scikit-learn's.
# To save running time, I run on a simulated dataset

from fast_gradient_algorithm import computegrad,obj,backtracking,fastgrad
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression

# Generate a simple simulated dataset
X = np.random.random((100,10))
y = np.concatenate((np.ones(67),-np.ones(33)),axis=0)

# Split data into a training set and a validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

def sk_result(lambd,x,y):
	'''
	This function implements regularized logistic regression and gets values of coeffcients and objective function

	:param lambd: Value of lambda
	:param x: Predictors
	:param y: Response
	:return: lr_coef: Coefficient of the features in the decision function
	:return: obj_sk: Value of objective function
	'''
	n = len(y)
	lr = LogisticRegression(penalty='l2',C=1/(2*lambd*n),fit_intercept=False,tol=10e-8,max_iter=1000)
	lr.fit(x,y)
	lr_coef = lr.coef_
	obj_sk = obj(beta=lr.coef_.flatten(),lambd=lambd,x=x,y=y)
	return (lr_coef,obj_sk)

def fast_result(beta,theta,lambd,x,y):
	'''
	This function implements my own fast gradient algorithm and gets values of coeffcients and objective function

	:param beta: Initial point
	:param theta: Initial theta
	:param lambd: Value of lambda
	:param x: Predictors
	:param y: Response
	:return: beta_fast: Coefficient of the features in the decision function
	:return: obj_fast: Value of objective function
	'''
	beta_fast = fastgrad(beta=beta,theta=theta,lambd=lambd,x=X_train,y=y_train)
	beta_last = beta_fast[-1,:]
	obj_fast = obj(beta=beta_last,lambd=lambd,x=x,y=y)
	return (beta_fast,obj_fast)


# Use sk_result and fast_result functions to run an experimental comparison 
# between scikit-learn's and my own fast gradient algorithm
# First, users should initialize parameters in these two functions
d = np.size(X,1)
beta = np.zeros(d)
theta = np.zeros(d)
lambd = 0.1

# Use the functions to get the results and then make comparisons
lr_coef,obj_sk = sk_result(lambd=lambd,x=X_train,y=y_train)
beta_fast,obj_fast = fast_result(beta=beta,theta=theta,lambd=lambd,x=X_train,y=y_train)

# Print the results to make comparisons
# Compare beta values
print(lr_coef)
print(beta_fast)

# Compare objective values
print(obj_sk)
print(obj_fast)