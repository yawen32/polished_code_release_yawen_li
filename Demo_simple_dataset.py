# This demo file allows users to launch fast gradient algorithm to solve logistic regression 
# problem on a simple simulated dataset

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

# Initialize parameters in fast gradient algorithm
d = np.size(X,1)
beta = np.zeros(d)
theta = np.zeros(d)
lambd = 0.1

# Implement fast gradient algorithm
beta_fast = fastgrad(beta=beta,theta=theta,lambd=lambd,x=X_train,y=y_train)
 
# Print the most recent values in the last row
print(beta_fast[-1,:])