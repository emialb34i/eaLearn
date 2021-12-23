import numpy as np

"""

Implementation of linear regression algortihms.

    1. LinearRegression - basic linear regression (OLS)
    2. Lasso - linear regression with l1 regularization
    3. Ridge - linear regression with l2 regularization
    4. ElasticNet - linear regression with l1 AND l2 regularizations

"""

class Regression:

    def __init__():
        pass



class LinearRegression:

    def fit(self, X, y, fit_intercept=True):
       
        # if we data does not have intercept we add it
        if fit_intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis=1)

        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        
        # estimate params
        XtX = np.dot(self.X.T, self.X)
        XtX_inverse = np.linalg.inv(XtX)
        Xty = np.dot(self.X.T, self.y)
        self.beta_hats = np.dot(XtX_inverse, Xty)

        # make_predictions
        self.y_hat = np.dot(self.X, self.beta_hats)


    def predict(self, X_test, fit_intercept=True):
        # insert intercept
        if fit_intercept:
            X_test = np.insert(X_test, 0, 1, axis=1)
        self.y_test_hat = np.dot(X_test, self.beta_hats)
        return self.y_test_hat



