import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).sum() / len(y_true)

def multivariate_normal(x, mu, cov):
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))