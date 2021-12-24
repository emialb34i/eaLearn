import numpy as np

"""

Implementation of linear regression algortihms.

    1. LinearRegression - basic linear regression (OLS)
    2. Lasso - linear regression with l1 regularization
    3. Ridge - linear regression with l2 regularization
    4. ElasticNet - linear regression with l1 AND l2 regularizations

"""

class Regression:

    def __init__(self, alpha=0.001, fit_intercept=True, max_iter=100):
        self.max_iter = max_iter
        self.alpha = alpha
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y):
        # if we data does not have intercept we add it
        if self.fit_intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis=1)
        
        # init betas randomly between [-1/N, 1/N]
        N = X.shape[1]
        self.beta_hats = np.random.uniform(-1/N, 1/N, size=(N, ))

        # gradient descent on squared loss
        self.training_errors = []
        for i in range(self.max_iter):
            y_pred = self.beta_hats@self.X
            # mean squared error
            mse = np.mean(0.5*(y_pred-y)**2)
            self.training_errors.append(mse)
            # grad of mse wrt to betas
            grad = ((y_pred-y)@X)/len(y)
            # gradient descent step
            self.beta_hats = self.beta_hats - self.alpha*grad

    def predict(self, X):
        # insert intercept
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        self.y_pred = X@self.beta_hats
        return self.y_pred



class LinearRegression(Regression):

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        pass

    def fit(self, X, y):
       
        # if we data does not have intercept we add it
        if self.fit_intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis=1)

        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        
        # estimate params (closeed form)
        XtX_inverse = np.linalg.inv(self.X.T@self.X)
        Xty = self.X.T@self.y
        self.beta_hats = XtX_inverse@Xty

        # make_predictions
        self.y_hat = np.dot(self.X, self.beta_hats)


    def predict(self, X):
        return super(LinearRegression, self).predict(X)


class Lasso(Regression):
    # TODO: add L1 regularaziation
    pass
        



