# This demo file allows users to launch fast gradient algorithm to solve logistic regression 
# problem on a real dataset: spam
# The dataset comes from: https://statweb.stanford.edu/~tibs/ElemStatLearn/b

from fast_gradient_algorithm import computegrad,obj,backtracking,fastgrad
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression

# Download dataset
spam = pd.read_csv('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header = None)
X = np.asarray(spam)[:,0:-1]
y = spam[spam.columns[-1]]
y = y.replace(0,-1)

# Split data into a training set and a validation set
ind = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest',sep=' ', header = None)
ind = np.array(ind).T[0]
X_train = X[ind == 0,:]
X_test = X[ind == 1,:]
y_train = y[ind == 0]
y_test = y[ind == 1]
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Initialize parameters in fast gradient algorithm
d = np.size(X,1)
beta = np.zeros(d)
theta = np.zeros(d)
lambd = 0.1

# Implement fast gradient algorithm
beta_fast = fastgrad(beta=beta,theta=theta,lambd=lambd,x=X_train,y=y_train)
# Print the most recent values in the last row
print(beta_fast[-1,:])