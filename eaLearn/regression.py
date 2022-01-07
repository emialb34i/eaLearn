import numpy as np

"""

Implementation of linear regression algortihms.

    1. LinearRegression - basic linear regression (OLS)
    2. Lasso - linear regression with l1 regularization
    3. Ridge - linear regression with l2 regularization
    4. ElasticNet - linear regression with l1 AND l2 regularizations

"""

class l1_regularization:

    def __init__(self, lam):
        self.lam = lam

    def __call__(self, w):
        return np.linalg.norm(w, ord=1) * self.lam

    def grad(self, w):
        return self.lam * np.sign(w)

class l2_regularization:

    def __init__(self, lam):
        self.lam = lam
    
    def __call__(self, w):
        return 0.5 * self.lam * np.linalg.norm(w, ord=2) 
    
    def grad(self, w):
        return self.lam * w

class l1_l2_regularization:

    def __init__(self, lam):
        self.lam = lam

    def __call__(self, w):
        l1 = np.linalg.norm(w, ord=1)
        l2 = 0.5 * np.linalg.norm(w, ord=2) 
        return self.lam * (l1 + l2)
    
    def grad(self, w):
        l1_grad = np.sign(w)
        l2_grad = self.lam * w
        return self.lam * (l1_grad + l2_grad)
   

class Regression:

    def __init__(self, learning_rate=0.01, fit_intercept=True, max_iter=1000):
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
    
    def fit(self, X, y):
        # if we data does not have intercept we add it
        if self.fit_intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis=1)

        # m training examples, n features
        m, n = X.shape
        
        # init weights randomly
        self.w = np.random.rand(n)

        # gradient descent on squared loss
        self.training_errors = []
        for _ in range(self.max_iter):
            y_pred = X@self.w
            # mean squared error
            mse = np.mean((y-y_pred)**2) + self.regulirization(self.w)
            self.training_errors.append(mse)
            # grad of mse wrt to w
            grad_w = -((y-y_pred)@X)*(2/m) + self.regulirization.grad(self.w)
            # gradient descent step
            self.w = self.w - self.learning_rate*grad_w
        
        return self

    def predict(self, X):
        # insert intercept
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        y_pred = X@self.w
        return y_pred



class LinearRegression(Regression):

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        pass

    def fit(self, X, y):
       
        # if the data does not have intercept we add it
        if self.fit_intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)
            X = np.concatenate((ones, X), axis=1)
        
        # estimate params (closed form)
        XtX_inverse = np.linalg.inv(X.T@X)
        Xty = X.T@y
        self.w = XtX_inverse@Xty

    def predict(self, X):
        return super(LinearRegression, self).predict(X)


class Lasso(Regression):

    def __init__(self, lam=1.0, learning_rate=0.01, fit_intercept=True, max_iter=1500):
        self.regulirization = l1_regularization(lam)
        super(Lasso, self).__init__(learning_rate, fit_intercept, max_iter)

class Ridge(Regression):

    def __init__(self, lam=1.0, learning_rate=0.01, fit_intercept=True, max_iter=1500):
        self.regulirization = l2_regularization(lam)
        super(Ridge, self).__init__(learning_rate, fit_intercept, max_iter)

class ElasticNet(Regression):

    def __init__(self, lam=1.0, learning_rate=0.01, fit_intercept=True, max_iter=1500):
        self.regulirization = l1_l2_regularization(lam)
        super(ElasticNet, self).__init__(learning_rate, fit_intercept, max_iter)
    
        



