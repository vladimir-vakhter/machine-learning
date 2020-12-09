import math
import numpy as np
#-------------------------------------------------------------------------
'''
    Problem: k nearest neighbor (KNN)
    In this problem, you will implement a classification method using k nearest neighbors. 
    The main goal of this problem is to get familiar with the basic settings of classification problems. 
    KNN is a simple method for classification problems.
    You could test the correctness of your code by typing `nosetests test.py` in the terminal.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submiting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your dropbox automatically sychronize your solution between your home computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework, build your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other people's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other student about this homework, only discuss high-level ideas or using psudo-code. Don't discuss about the solution at the code level. For example, discussing with another student about the solution of a function (which needs 5 lines of code to solve), and then working on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences  (like changing variable names) will violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Historical Data: in one year, we ended up finding 25% of the students in the class violating this term in their homework submissions and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #########################################
    ## CHANGE CODE HERE
    Read_and_Agree = True
    #########################################
    return Read_and_Agree
 
#--------------------------
def compute_distance(Xtrain, Xtest):
    '''
        compute the Euclidean distance between instances in a test set and a training set 
        Input:
            Xtrain: the feature matrix of the training dataset, a float python matrix of shape (n_train by p).
                    Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Xtest:  the feature matrix of the test dataset, a float python matrix of shape (n_test by p).
                    Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
        Output:
            D:      the distance between instances in Xtest and Xtrain, a float python matrix of shape (ntest, ntrain),
                    the (i,j)-th element of D represents the Euclidean distance between the i-th instance in Xtest and j-th instance in Xtrain.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    
    #the dimensions of both the train and the test datasets
    n_train, p = Xtrain.shape
    n_test = Xtest.shape[0]
    #the list of the distances between Xtest[i] and Xtrain[j]
    distances = []
    #for each i-th test sample
    for i in range(0, n_test):              
        #calculate the distance to each j-th train sample
        for j in range(0, n_train):         
            #the initial value of the distance
            distance = 0.0
            #go through all the features
            for k in range(0, p):
                 distance = distance + math.pow((Xtest[i][k] - Xtrain[j][k]), 2)
            distances.append(math.sqrt(distance))
    #create the matrix of the distances
    D = np.array(distances, dtype=float).reshape((n_test, n_train))
    
    #########################################
    return D 

#-------------------------
def maxFreqElement(l):
    '''
        find the most frequent element in the list
    '''
    res = l[0] 
    max_freq = 0
    for i in l: 
        freq = l.count(i) 
        if freq > max_freq: 
            max_freq = freq 
            res = i 
    return res

#--------------------------
def k_nearest_neighbor(Xtrain, Ytrain, Xtest, K = 3):
    '''
        compute the labels of test data using the K nearest neighbor classifier.
        Input:
            Xtrain: the feature matrix of the training dataset, a float numpy matrix of shape (n_train by p).
                    Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Ytrain: the label vector of the training dataset, an integer python list of length n_train.
                    Each element in the list represents the label of the training instance.
                    The values can be 0, ..., or num_class-1. num_class is the number of classes in the dataset.
            Xtest:  the feature matrix of the test dataset, a float python matrix of shape (n_test by p).
                    Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            K:      the number of neighbors to consider for classification.
        Output:
            Ytest: the predicted labels of test data, an integer numpy vector of length ntest.
        Note: you cannot use any existing package for KNN classifier.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    
    #the dimensions of both the train and the test datasets
    n_train, p = Xtrain.shape
    n_test = Xtest.shape[0] 
    #compute the distances
    D = compute_distance(Xtrain, Xtest)
    #the list of the labels for all the test samples
    labels = []
    #for each sample in the test dataset
    for i in range(0, n_test):              
        #the indices of K the shortest distances
        indices = D[i].argsort()[:K]
        #the labels of the train samples that are the nearest to the current test sample
        nearest_labels = []
        for j in range(0, indices.size):
            nearest_labels.append(Ytrain[indices[j]])
        labels.append(maxFreqElement(nearest_labels))
    #create the matrix of the labels for the test samples
    Ytest = np.array(labels, dtype=int).reshape((n_test,))
    
    #########################################
    return Ytest 
