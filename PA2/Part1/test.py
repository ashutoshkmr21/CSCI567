from data_loader import data_processing_linear_regression
from linear_regression import linear_regression_invertible
import numpy as np


filename = 'winequality-white.csv'
# Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing_linear_regression(filename, True, False, 0)
# w = linear_regression_invertible(Xtrain, ytrain)

# print('w is ', w)


from linear_regression import mapping_data

print(mapping_data(np.array([[1,2,3],[2,2,1]]), 3))
