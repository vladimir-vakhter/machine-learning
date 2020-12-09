#!/usr/bin/python3				
# -*- coding: utf-8 -*-	

import math
import numpy as np
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: Linear Regression (Maximum Likelihood)
    In this problem, you will implement the linear regression method based upon maximum likelihood (least square).
    w'x + b = y
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
    Note: please don't use any existing package for linear regression problem, implement your own version.
''' 

#--------------------------
def compute_Phi(x,p):
    '''
        Let's start with a simple dataset. This dataset (polynomial curve fitting) has been used in many machine learning textbooks.
        In this dataset, we create a feature matrix by constructing multiple features from one single feature.
        For example, suppose we have a data example, with an input feature of value 2., we can create a 4-dimensional feature vector (when p=4) 
        using the different polynoials of the input like this: ( 1., 2., 4., 8.). 
        Here the first dimension is the 0-th polynoial of 2., 2^0 = 1.
        The second dimension is the 1st polynoial of 2., 2^1 = 2
        The third dimension is the 2nd polynoial of 2., 2^2 = 4
        The third dimension is the 3rd polynoial of 2., 2^3 = 8
        Now in this function, x is a vector, containing multiple data samples. For example, x = [2,3,5,4] (4 data examples)
        Then, if we want to create a 3-dimensional feature vector (when p=3) for each of these examples, then we have a feature matrix Phi (4 by 3 matrix).
        [[1, 2,  4],
         [1, 3,  9],
         [1, 4, 16],
         [1, 5, 25]]
        In this function , we need to compute the feature matrix (or design matrix) Phi from x for polynoial curve fitting problem. 
        We will construct p polynoials of x as the p features of the data samples. 
        The features of each sample, is x^0, x^1, x^2 ... x^(p-1)
        Input:
            x : a vector of samples in one dimensional space, a numpy vector of shape n by 1.
                Here n is the number of samples.
            p : the number of polynomials/features
        Output:
            Phi: the design/feature matrix of x, a numpy matrix of shape (n by p).
                 The i-th column of Phi represent the i-th polynomials of x. 
                 For example, Phi[i,j] should be x[i] to the power of j.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # number of samples
    n = x.size
    # list of polynomials (empty)
    l = []
    # calculate (p-1) polynomials for each of n samples	
    for i in range(0, n):
        for j in range(0, p):
            curr_val = math.pow(x.item(i), j)
            l.append(curr_val)
    # feature matrix
    Phi = np.asmatrix(np.array(l).reshape((n, p)))
    #########################################
    return Phi 

#--------------------------
def least_square(Phi, y):
    '''
        Fit a linear model on training samples. Compute the paramter w using Maximum likelihood (equal to least square).
        Input:
            Phi: the design/feature matrix of the training samples, a numpy matrix of shape n by p
                Here n is the number of training samples, p is the number of features
            y : the sample labels, a numpy vector of shape n by 1.
        Output:
            w: the weights of the linear regression model, a numpy float vector of shape p by 1. 
        Hint: you could use np.linalg.inv() to compute the inverse of a matrix
              The problem can be solved using 1 line of code.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # a vector of the weights of the linear regression model
    w = np.linalg.inv(Phi.T * Phi) * Phi.T * y
    #########################################
    return w 

#--------------------------
def ridge_regression(Phi, y, alpha=0.001):
    '''
        Fit a linear model on training samples. Compute the paramter w using Maximum posterior (equal to least square with L2 regularization).
        min_w sum_i (y_i - Phi_i * w)^2/2 + alpha * w^T * w 
        Input:
            Phi: the design/feature matrix of the training samples, a numpy matrix of shape n by p
                Here n is the number of training samples, p is the number of features
            y : the sample labels, a numpy vector of shape n by 1.
            alpha: the weight of the L2 regularization term, a float scalar.
        Output:
            w: the weights of the linear regression model, a numpy float vector of shape p by 1. 
        Hint: you could use np.linalg.inv() to compute the inverse of a matrix. 
              The problem can be solved using 2 lines of code.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
	# identity matrix
    identity_matrix = np.eye(Phi.shape[1], dtype = int)
    # a vector of the weights of the linear regression model after regularization
    w = np.linalg.inv((Phi.T * Phi) + alpha * identity_matrix) * Phi.T * y
    #########################################
    return w 

