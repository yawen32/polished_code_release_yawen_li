import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.linear_model import LogisticRegression

def computegrad(beta, lambd, x, y):
	'''
	This function computes the gradient of objective function.

	:param beta: Current point
	:param lambd: Value of lambda
	:param x: Predictors
	:param y: Response
	:return: The gradient of objective function

	'''
	n = x.shape[0]
	yx = y[:, np.newaxis]*x
	yxb = yx.dot(beta[:,np.newaxis])
	return -1/n*np.sum(yx*np.exp(-yxb)/(1+np.exp(-yx.dot(beta)))[:,np.newaxis],axis=0)+2*lambd*beta


def obj(beta, lambd, x, y):
	'''
	This function computes objective function of logistic regression problem

	:param beta: Current point
	:param lambd: Value of lambda
	:param x: Predictors
	:param y: Response
	:return:  Objective function of logistic regression problem

	'''
	n = x.shape[0]
	yxb = y*x.dot(beta)
	return 1/n*np.sum(np.log(1+np.exp(-yxb)))+lambd*np.linalg.norm(beta)**2


def backtracking(beta, lambd, x, y, t=1, alpha=0.5, decrease = 0.8, max_iter=100):
	'''
	This function implements the backtracking rule.

	:param beta: Current point
	:param lambd: Value of lambda
	:param x: Predictors
	:param y: Response
	:param t: Starting (maximum) step size
	:param alpha: Constant used to define sufficient decrease condition
	:param decrease: Constant used to decrease t if the previous t doesn't work
	:param max_iter: Maximum number of iterations
	:return: t: Updated step size
	'''
	grad_beta = computegrad(beta=beta,lambd=lambd,x=x,y=y)
	norm_grad_beta = np.linalg.norm(grad_beta)
	found_t = 0
	itera = 0
	while found_t == 0 and itera < max_iter:
		if obj(beta=beta-t*grad_beta,lambd=lambd,x=x,y=y) < (obj(beta=beta,lambd=lambd,x=x,y=y)-alpha*t*norm_grad_beta**2):
			found_t = 1
		else:
			t = t*decrease
			itera = itera + 1
		return t


def fastgrad(beta, theta, lambd, x, y, t=1, max_iter=100):
	'''
	This function implements the my own fast gradient algorithm with backtracking rule.

	:param beta: Initial point
	:param theta: Initial theta
	:param lambd: Value of lambda
	:param x: Predictors
	:param y: Response
	:param t: Starting (maximum) step size
	:param max_iter: Maximum number of iterations
	:return: theta_vals: Matrix of theta values at each iteration, with the most recent values in the last row
	'''
	beta = beta
	theta = theta
	theta_vals = theta
	itera = 0
	while(itera < max_iter):
		t = backtracking(beta=beta,lambd=lambd,x=x,y=y)
		beta_new = theta - t*computegrad(beta=theta,lambd=lambd,x=x,y=y)
		theta = beta_new + (itera/(itera+3))*(beta_new-beta)
		theta_vals = np.vstack((theta_vals,theta))
		beta = beta_new
		itera += 1
	return theta_vals