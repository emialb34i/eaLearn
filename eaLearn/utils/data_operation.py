import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).sum() / len(y_true)
