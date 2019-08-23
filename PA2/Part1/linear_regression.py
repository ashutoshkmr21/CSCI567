"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = None
    try:
        err = ((np.dot(X,w)-y)**2).mean()
    except Exception as e:
        pass

    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  w = None
  try:
      w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
  except:
      pass
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    w = None
    try:
        mat = np.dot(X.T, X)
        mat = is_invertible(mat)
        w = np.dot(np.linalg.inv(mat), np.dot(X.T, y))
    except Exception as e:
        pass
    return w

def is_invertible(a, k=None):
    if k == None:
        k = 0.1
    while True:
        eig_vals = np.linalg.eigvals(a)
        if 0.00001 > abs(min(eig_vals, key=abs)):
            a = a + (k*np.identity(a.shape[0]))
        else:
            return a
    return a

###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    w = None
    try:
        mat = np.dot(X.T, X)
        mat = mat + (lambd*np.identity(mat.shape[0]))
        # mat = is_invertible(mat, lambd)

        w = np.dot(np.linalg.inv(mat), np.dot(X.T, y))

    except Exception as e:
        pass
    print('lambda is ', lambd, ' w is ', w)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    initial_exp, final_exp = -19, 20
    min_mse = 1e99
    best_index = None
    for i in range(initial_exp, final_exp):
        w = regularized_linear_regression(Xtrain, ytrain, 10**i)
        # print('W   is ', w)
        mse = mean_square_error(w, Xval, yval)
        print('mse is ', mse, ' min_mse is ', min_mse, ' index is ', i, '  power ', 10**i)
        if mse < min_mse:
            min_mse = mse
            best_index = i
            print('Best Index', best_index)
    return 10**best_index
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################

    # ind = X.shape[1]
    x_c = X.tolist()
    x_c = np.array(x_c)
    for k in range(2, power+1):
        # X = np.insert(X, ind+i, X[:,i]**, axis=1)
        X = np.append(X, x_c**k, axis=1)

    return X


